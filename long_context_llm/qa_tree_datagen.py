import os, random
import argparse
from types import SimpleNamespace

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, Dataset

from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from long_context_llm.scripts import BestOfNSampler
from HELMET.model_utils import tokenize
from data import load_data, load_train_data
from scripts.prompts import SYSTEM_PROMPTS, CLARIFY_QUESTIONS, TERMINATION_CHECKS

import torch, transformers
import copy, json, pickle
from vllm import LLM, SamplingParams
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


# Set up argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_sample_size", type=int, default=1)
    parser.add_argument("--max_tree_depth", type=int, default=1)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--parent_dataset_name_or_path", type=str, default=None)
    return parser.parse_args()

def replace_last_occurrence(text, old, new):
    parts = text.rsplit(old, 1)
    return new.join(parts)

# callable that takes a list of raw text and returns a list of corresponding reward scores
def queries_to_scores(response_lst, prompt):
    reward_lst = []
    for response in response_lst:
        reward=0
        reward_lst.append(reward)
    return reward_lst

def chat_template_format(question, tokenizer, conversations=[], system_prompt=None):
    # conversations is a list of dictionaries with keys "role" and "content"
    # example: [{"role": "user", "content": "Hello!"}, {"role": "system", "content": "Hi!"}]
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversations)
    messages.append({"role": "user", "content": question})
    output = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return output

def context_retrieve(sampler, context_pieces, new_question, tokenizer, system_prompt=None, sampling_params=None):
    new_context = []
    context_check_questions = []
    for context_piece in context_pieces:
        context_check_question = f"Question:{new_question}\n\nContext:{context_piece}\n\nDoes the given piece of context help answer the question in any ways? Answer 'yes' or 'no' or 'maybe' at the end of your response!" #Answer only with yes or no!
        context_check_question = chat_template_format(context_check_question, tokenizer=tokenizer, system_prompt=system_prompt)
        context_check_questions.append(context_check_question)
    context_check_responses = sampler.generate(queries=context_check_questions, sampling_params=sampling_params)
    for ctr, response in enumerate(context_check_responses):
        if "no" in response.outputs[0].text.lower().split("answer:")[-1]:
            continue
        new_context.append(context_pieces[ctr])
    new_context = "...\n\n...".join(new_context)
    return new_context


def batched_context_retrieve(sampler, context_pieces, new_questions, tokenizer, system_prompt=None, sampling_params=None):
    context_check_questions = []
    outputs, marker_outputs = [], []
    for new_question in new_questions:
        for context_piece in context_pieces:
            context_check_question = f"Question:{new_question}\n\nContext:{context_piece}\n\nDoes the given piece of context help answer the question in any ways? Answer 'yes' or 'no' or 'maybe' at the end of your response! Follow the format: [reasoning] + Answer: (yes/maybe/no). Be concise." #Answer only with yes or no!
            context_check_question = chat_template_format(context_check_question, tokenizer=tokenizer, system_prompt=system_prompt)
            context_check_questions.append(context_check_question)
    context_check_responses = sampler.generate(queries=context_check_questions, sampling_params=sampling_params)
    new_context = []
    new_context_marker = []
    
    for ctr, response in enumerate(context_check_responses):
        if "no" in response.outputs[0].text.lower().split("answer:")[-1]:
            pass
        else:
            new_context.append(context_pieces[ctr % len(context_pieces)])
            new_context_marker.append("<para " + str(ctr % len(context_pieces)) + ">")
        if (ctr+1) % len(context_pieces) == 0:
            print(len(context_pieces), len(new_context))
            if len(new_context) == 0:
                outputs.append("No context was found to be helpful for the question.")
                marker_outputs.append("No context was found to be helpful for the question.")
            else:
                outputs.append("...\n\n...".join(new_context))
                marker_outputs.append("The following pieces of context were found to be helpful for the question:\n" + " ".join(new_context_marker))
            new_context = []
            new_context_marker = []
    return outputs, marker_outputs


def mark_context(context_pieces):
    # put the markers in between the context pieces
    # <para 0> ... </para 0> <para 1> ... </para 1> ...
    marked_context = ""
    for idx, context_piece in enumerate(context_pieces):
        marked_context += f"<para {idx}> {context_piece} </para {idx}>"
    return marked_context

class Reasoning(BaseModel):
    explanation: str
    final_answer: str

def gpt_answer_check(question, ground_truth, answer):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You help verify if the answers are correct. The prediction might be longer but you will classify it as correct as long as it contains the same semantic meaning as the ground truth."},
            {
                "role": "user",
                "content": f"Question: '{question}'. Ground truth: '{ground_truth}'. Prediction: '{answer}'. Help me verify if the prediction answers the question correctly. Format: reply 'Yes' or 'No' as the final answer."
            }
        ],
        response_format=Reasoning
    )

    correct = int("yes" in completion.choices[0].message.parsed.final_answer.lower().split(" "))
    return correct

class RandomPrompt:
    def __init__(self, prompts):
        self.prompts = prompts

    def __call__(self):
        # When the instance is called like a function, return a random prompt
        return random.choice(self.prompts)

    def __str__(self):
        # When converted to a string (e.g., print), also return a random prompt
        return random.choice(self.prompts)



class qa_node:
    # class that records the question-answer pairs and the context that is used to answer the question
    # it also records the performance of the answer
    def __init__(self, question, local_context, context_pieces, clarify_question, stop_question, sampling_size=16):
        self.question = question
        self.local_context = local_context
        self.complete_context_pieces = context_pieces
        self.clarify_question = clarify_question
        self.stop_question = stop_question
        self.sampling_size = sampling_size

        self.answer = None
        self.inter_question = None
        self.inter_answer = None
        self.performance = None
        self.termination = None
        self.conversation = None
        self.children = []
        self.parent = None 
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def add_answer(self, inter_question, inter_answer, answer, performance, termination):
        self.inter_question = inter_question
        self.inter_answer = inter_answer
        self.answer = answer
        self.performance = performance
        self.termination = termination
    
    def add_conversation(self, conversation):
        self.conversation = conversation

    def get_path(self):
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        # reverse the path from root to leaf
        path.reverse()
        return path

    def depth(self):
        node = self
        depth = 0
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def sampling_step(self, qa_dataset, data, sampler, tokenizer, system_prompt, sampling_params_context_check, sampling_params_question_answer):
        # gather QA path history
        conversations = []
        if self.conversation is not None:
            # ignore the system prompt and the main context
            conversations = self.conversation[3:]
        clarify_question = self.clarify_question()
        stop_question = self.stop_question()
        local_system_prompt = system_prompt()
        # sample new questions
        prompt = chat_template_format(self.local_context + "\n" + clarify_question, tokenizer=tokenizer, system_prompt=local_system_prompt, conversations=conversations)
        generated_questions = sampler.generate(queries=[prompt])
        new_questions = [generated_questions[0].outputs[idx].text for idx in range(len(generated_questions[0].outputs))]

        # gather new context
        new_contexts, new_marks = batched_context_retrieve(sampler, self.complete_context_pieces, new_questions, tokenizer=tokenizer, system_prompt=local_system_prompt, sampling_params=sampling_params_context_check)
        
        omit_context = "[Ommited context that you have seen]"
        intermediate_prompts = [chat_template_format("Answer the question you asked with the given context [{new_context}], summarize your findings as well." + new_context, tokenizer=tokenizer, system_prompt=local_system_prompt, conversations=conversations + [{"role": "user", "content": omit_context + "\n" + clarify_question}, {"role": "assistant", "content": new_question}]) for new_context, new_question in zip(new_contexts, new_questions)]

        new_answers = sampler.generate(queries=intermediate_prompts, sampling_params=sampling_params_question_answer)
        new_answers = [new_answers[idx].outputs[0].text for idx in range(len(new_answers))]

        # generate final answers
        new_prompts = [chat_template_format("Now, let's answer the final question, be concise in your answer." + data['question'], tokenizer=tokenizer, system_prompt=local_system_prompt, conversations=conversations + [{"role": "user", "content": omit_context + "\n" + clarify_question}, {"role": "assistant", "content": new_question}, {"role": "user", "content": f"Related context are given:\n {new_context}"}, {"role": "assistant", "content": new_answer}]) for new_context, new_question, new_answer in zip(new_contexts, new_questions, new_answers)]

        final_answers = sampler.generate(queries=new_prompts, sampling_params=sampling_params_question_answer)
        final_answers = [final_answers[idx].outputs[0].text for idx in range(len(final_answers))]

        # termination check, have you answered the question correctly?
        termination_prompts = [chat_template_format(stop_question, tokenizer=tokenizer, system_prompt=local_system_prompt, conversations=conversations + [{"role": "user", "content": omit_context + "\n" + clarify_question}, {"role": "assistant", "content": new_question}, {"role": "user", "content": f"Related context are given:\n {new_context}"}, {"role": "assistant", "content": new_answer}, {"role": "user", "content": "Now, let's answer the final question, be concise in your answer." + data['question']}, {"role": "assistant", "content": final_answer}]) for new_context, new_question, new_answer, final_answer in zip(new_contexts, new_questions, new_answers, final_answers)]
        termination_checks_json = sampler.generate(queries=termination_prompts, sampling_params=sampling_params_context_check)

        termination_checks = [termination_checks_json[idx].outputs[0].text for idx in range(len(termination_checks_json))]

        # evaluate the performance
        final_scores = []
        for answer in final_answers:
            mets, others = qa_dataset['post_process']({"output": answer}, data)
            mets['rougeLsum_f1'] = float(mets['rougeLsum_f1']) + gpt_answer_check(question=data['question'], ground_truth=data['answer'], answer=answer)
            final_scores.append(mets['rougeLsum_f1'])
        
        # create the children nodes
        for new_question, new_context, new_mark, new_answer, final_answer, final_score, termination_check in zip(new_questions, new_contexts, new_marks, new_answers, final_answers, final_scores, termination_checks):
            new_node = qa_node(new_question, new_context, self.complete_context_pieces, self.clarify_question, self.stop_question, self.sampling_size)
            new_node.add_answer(new_question, new_answer, final_answer, final_score, termination_check)

            # Currently without termination check
            new_conversation = [{"role": "system", "content": local_system_prompt}] + self.conversation[1:] + [{"role": "user", "content": clarify_question}, {"role": "assistant", "content": new_question}, {"role": "user", "content": "Help me find relevant context to answer the previous clarifying quesiton."}, {"role": "assistant", "content": new_mark}, {"role": "user", "content": "Based on the relevant context, answer the previous clarifying question."}, {"role": "assistant", "content": new_answer}, {"role": "user", "content": "Now, let's answer the final question, be concise in your answer." + data['question']}, {"role": "assistant", "content": final_answer}]
            new_node.add_conversation(new_conversation)
            self.add_child(new_node)

        return 
    
    def bfs_sampling(self, qa_dataset, data, sampler, tokenizer, system_prompt, sampling_params_context_check, sampling_params_question_answer, sampling_depth=1):
        queue = [self]
        while len(queue) > 0:
            node = queue.pop(0)
            if node.depth() <= sampling_depth:
                node.sampling_step(qa_dataset, data, sampler, tokenizer, system_prompt, sampling_params_context_check, sampling_params_question_answer)
                print(f"Node depth: {node.depth()}, number of children: {len(node.children)}")
                queue.extend(node.children)
        return

    def bfs_traversal(self):
        queue = [self]
        output = []
        while len(queue) > 0:
            node = queue.pop(0)
            output.append(node)
            queue.extend(node.children)
        return output



def main():
    args = parse_args()

    # Get the arguments
    base_model_name = args.model_name_or_path
    MAX_SAMPLE_SIZE = args.max_sample_size
    MAX_TREE_DEPTH = args.max_tree_depth
    dataset_name = args.dataset_name

    config = AutoConfig.from_pretrained(base_model_name)
    model = LLM(model=base_model_name, tensor_parallel_size=8, max_seq_len_to_capture=131072, enable_chunked_prefill=True, max_num_batched_tokens=4096)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, n=MAX_SAMPLE_SIZE, max_tokens=512) #best_of=256, stop_token_ids=tokenizer.encode("?", add_special_tokens=False)
    sampling_params_context_check = SamplingParams(temperature=0.2, top_p=0.95, n=1, max_tokens=128) 
    sampling_params_question_answer = SamplingParams(temperature=0.2, top_p=0.95, n=1, max_tokens=1024)
        
    best_of_n = BestOfNSampler(model, tokenizer, queries_to_scores, sampling_params=sampling_params)

    data_args = {
        "max_test_samples": None,
        "shots": 0,
        "seed": 42,
        "generation_max_length": 100,
        "input_max_length": 128000,
        }
    data_args = SimpleNamespace(**data_args)
    qa_dataset = load_train_data(data_args, dataset_name)
    system_prompt = RandomPrompt(SYSTEM_PROMPTS)
    stop_question = RandomPrompt(TERMINATION_CHECKS)

    perform_improv = []
    ids, seq_lens = [], []
    best_path_json, worst_path_json = [], []
    best_scores, worst_scores = [], []
    questions, answers = [], []

    # Having a reloading mechanism for the dataset
    completed_dataset = None
    if args.parent_dataset_name_or_path is None:
        pass
    else:
        completed_dataset = load_dataset(args.parent_dataset_name_or_path)['train']

    # if the dataset is already completed, we will skip the data points that are already completed
    if completed_dataset is not None:
        ids = completed_dataset['id']
        seq_lens = completed_dataset['seq_len']
        best_path_json = completed_dataset['chosen']
        worst_path_json = completed_dataset['rejected']
        best_scores = completed_dataset['score_chosen']
        worst_scores = completed_dataset['score_rejected']
        questions = completed_dataset['question']
        answers = completed_dataset['answer']
    
    for data_ctr in range(len(qa_dataset['data'])):
        if data_ctr % args.world_size != args.local_rank or data_ctr in ids:
            # for distributed inference
            continue
        data = qa_dataset['data'][data_ctr]
        # context split for every 512 tokens
        tokenized_context = tokenizer(data['context'], return_tensors="pt")
        context_len = 512
        context_pieces = [tokenizer.decode(tokenized_context['input_ids'][0][idx:idx+context_len], skip_special_tokens=True) for idx in range(0, len(tokenized_context['input_ids'][0]), context_len)]
        local_system_prompt = system_prompt()
        # initial gatherings
        marked_context = mark_context(context_pieces)
        first_context, first_mark = batched_context_retrieve(best_of_n, context_pieces, [data["question"]], tokenizer=tokenizer, system_prompt=local_system_prompt, sampling_params=sampling_params_context_check)
        first_context, first_mark = first_context[0], first_mark[0]
        clarify_question = RandomPrompt(CLARIFY_QUESTIONS)
        clarify_question.prompts = [prompt.replace("QUESTION_PLACE_HOLDER", data['question']) for prompt in clarify_question.prompts]

        root_prompt = chat_template_format(first_context + "\n" + data['question'], tokenizer=tokenizer)
        base_answer = best_of_n.generate(queries=[root_prompt], sampling_params=sampling_params_question_answer)
        base_answer = base_answer[0].outputs[0].text
        base_termination = best_of_n.generate(queries=[chat_template_format(stop_question, tokenizer=tokenizer, system_prompt=local_system_prompt, conversations=[{"role": "user", "content": data['question']}, {"role": "assistant", "content": base_answer}])], sampling_params=sampling_params_context_check)
        base_termination_json = base_termination[0].outputs[0].text
        try:
            base_termination = json.loads(base_termination_json)['answer']
        except:
            base_termination = base_termination_json

        mets, others = qa_dataset['post_process']({"output": base_answer}, data)
        root_node = qa_node(data['question'], local_context=first_context, context_pieces=context_pieces, clarify_question=clarify_question, stop_question=stop_question)
        root_node.add_answer(data['question'], base_answer, base_answer, float(mets['rougeLsum_f1']), base_termination)
        root_node.add_conversation([{"role": "system", "content": local_system_prompt}, {"role": "user", "content": marked_context + "\n\n" + data['question']}, {"role": "assistant", "content": base_answer}])

        # we sample children nodes, and expand the tree accordingly
        root_node.bfs_sampling(qa_dataset, data, best_of_n, tokenizer, system_prompt, sampling_params_context_check, sampling_params_question_answer, sampling_depth=MAX_TREE_DEPTH)

        # we then collect the best performing path
        best_conv, worst_conv = [], []
        best_score = -1
        worst_score = 2
        for node in root_node.bfs_traversal():
            if node.performance > best_score and node.depth() > 1:
                best_score = node.performance
                best_conv = node.conversation
            if node.performance < worst_score and node.depth() > 1:
                worst_score = node.performance
                worst_conv = node.conversation

        ids.append(data_ctr)
        seq_lens.append(len(tokenized_context['input_ids'][0]))
        best_path_json.append(best_conv)
        worst_path_json.append(worst_conv)
        best_scores.append(best_score)
        worst_scores.append(worst_score)
        questions.append(data["question"])
        answers.append(data["answer"])


    # final dataset upload
    dataset_dpo = Dataset.from_dict({"id": ids, "chosen": best_path_json, "rejected": worst_path_json, "score_chosen": best_scores, "score_rejected": worst_scores, "question": questions, "answer": answers, "seq_len": seq_lens})
    dataset_dpo.push_to_hub("YOUR REPO")
    
if __name__ == "__main__":
    main()