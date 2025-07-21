<h1 align="center"> 📖 Agentic Long Context Understanding 📖 </h1>
<p align="center"> <b>Self-Taught Agentic Long Context Understanding</b>  (<a href="https://arxiv.org/abs/2502.15920">ACL 25</a>). 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
</p>  

<p align="center"> AgenticLU refines complex, long-context queries through self-clarifications and contextual grounding, enabling robust long-document understanding in a single pass.
</p>

## Installation Requirements
This codebase is largely based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [Helmet](https://github.com/princeton-nlp/HELMET), kudos to them.

The requirements are the same
```
pip install openrlhf
pip install -r ./HELMET/requirements.txt
```


## Dataset \& Model

Dataset for SFT and DPO is avaliable at [here](https://huggingface.co/datasets/yzhuang/Agentic-Long-Context-Understanding-QA)

Model is available at [here](https://huggingface.co/yzhuang/Llama-3.1-8B-Instruct-AgenticLU)

## Data Generation Pipeline

To generate traces with your custom model or dataset, follow the instructions:

1. Get an OpenAI API key and set it as your env variable
```
export OPENAI_API_KEY="your_api_key_here"
```

2. Edit the bash sript as you needed for base model, search width and depth
```
PYTHONPATH="./":"$PYTHONPATH" python ./long_context_llm/qa_tree_datagen.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --max_sample_size 8 \
    --max_tree_depth 2 \
    --dataset_name yzhuang/narrative_qa
```

3. The traces will be avaliable to you as ```dataset_dpo```, feel free to add this line to push to your huggingface account.
```
dataset_dpo.push_to_hub("YOUR REPO")
```

## Example Usage

We show the training script of AgenticLU at [sft script](bash_scripts/sft_8b.sh), [dpo script](bash_scripts/rlhf_8b.sh).

It is important to get [ring-attention](https://github.com/zhuzilin/ring-flash-attention) to work, as the inputs are extremely long and requires ring-attention and deepspeed for training.

Examples for inferencing with the agentic workflow can be found [here](HELMET/scripts/run_agents.sh), with baseline prompting [scripts](HELMET/scripts/run_prompting.sh) avaliable.


## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at y5zhuang@ucsd.edu.

## Citation

If you find our paper and code useful, please cite us:
```r
@misc{zhuang2025selftaughtagenticlongcontext,
      title={Self-Taught Agentic Long Context Understanding}, 
      author={Yufan Zhuang and Xiaodong Yu and Jialian Wu and Ximeng Sun and Ze Wang and Jiang Liu and Yusheng Su and Jingbo Shang and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2502.15920},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.15920}, 
}
```
