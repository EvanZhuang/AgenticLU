# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Optional, Union

import torch
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

from trl.core import set_seed
from trl.models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper


class BestOfNSampler:
    def __init__(
        self,
        model: PreTrainedModelWrapper,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        queries_to_scores: Callable[[List[str]], List[float]],
        seed: Optional[int] = None,
        sampling_params: Optional[dict] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        r"""
        Initialize the sampler for best-of-n generation

        Args:
            model (`PreTrainedModelWrapper`):
                The pretrained model to use for generation
            tokenizer (`PreTrainedTokenizer` or `PreTrainedTokenizerFast`):
                Tokenizer associated with the pretrained model
            queries_to_scores (`Callable[[List[str]], List[float]]`):
                Callable that takes a list of generated texts and returns the associated reward scores
            length_sampler (`Any`):
                Sampler used to sample the length of the generated text
            sample_size (`int`):
                Number of samples to generate for each query
            seed (`int`, *optional*):
                Random seed used to control generation
            n_candidates (`int`):
                Number of candidates to return for each query
            generation_config (`GenerationConfig`, *optional*):
                Generation config passed to the underlying model's `generate` method.
                See `GenerationConfig` (https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig) for more details
        """
        if seed is not None:
            set_seed(seed)

        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        # if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
        #     raise ValueError(
        #         f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
        #     )

        self.model = model
        self.tokenizer = tokenizer

        self.queries_to_scores = queries_to_scores
        self.sampling_params = sampling_params
        self.gen_config = generation_config

    def generate(
        self,
        # tokenized_query: Union[List[int], torch.Tensor, List[torch.Tensor], List[List[int]]],
        queries: List[Union[List[str], torch.Tensor]],
        skip_special_tokens: bool = True,
        sampling_params: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs,
    ) -> List[List[str]]:
        r"""
        Generate the best of n samples for input queries

        Args:
            queries: list of queries to generate samples for in string format
            skip_special_tokens (`bool`):
                Whether to remove the special tokens from the output
            device (`str` or `torch.device`, *optional*):
                The device on which the model will be loaded
            **generation_kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `generate` method.
                This is used to override generation config

        Returns:
            List[List[str]]: A list of lists of generated texts
        """
        # queries = None

        # if isinstance(tokenized_query, torch.Tensor) and tokenized_query.ndim == 1:
        #     queries = tokenized_query.unsqueeze(0)
        # elif isinstance(tokenized_query, List):
        #     element_type = type(tokenized_query[0])
        #     if element_type is int:
        #         queries = torch.tensor(tokenized_query).unsqueeze(0)
        #     elif element_type is torch.Tensor:
        #         queries = [tensor.reshape((1, -1)) for tensor in tokenized_query]
        #     else:
        #         queries = [torch.tensor(query).reshape((1, -1)) for query in tokenized_query]

        #queries_idx = [self.tokenizer(x, padding=False, truncation=False, return_tensors="pt")['input_ids'] for x in queries]

        # result = []
        session_params = None
        if sampling_params is not None:
            session_params = sampling_params
        else:
            session_params = self.sampling_params
        result = self.model.generate(queries, session_params)


        # for query_idx, query in zip(queries_idx, queries):
            
            # query_idx = query_idx.repeat((self.sample_size, 1))
            # output = self.model.generate(
            #     query_idx.to(device),
            #     max_new_tokens=self.length_sampler(),
            #     generation_config=self.gen_config,
            #     **generation_kwargs,
            # ).squeeze()
            # output = self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
            # scores = torch.tensor(self.queries_to_scores(output, query))
            # output = [output[i] for i in scores.topk(self.n_candidates).indices]
            # result.append(output)

        return result