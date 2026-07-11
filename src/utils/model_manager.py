import os

import torch
import gc
from typing import Literal
import random
import time

from langchain_ollama import OllamaLLM
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    def __init__(self, model_id: str, device: str, backend: Literal["ollama", "huggingface"]):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.backend = backend

    def __enter__(self):
        if self.backend == "huggingface":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                              torch_dtype=torch.bfloat16,
                                                              low_cpu_mem_usage=True,
                                                              trust_remote_code=True,
                                                              device_map="auto",
                                                              offload_folder="offload"
                                                              )
            print(f"Memory Allocated after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, device_map="auto")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.backend == "ollama":
            self.model = OllamaLLM(model=self.model_id, temperature=0.8, num_predict=300)
        elif (self.backend == "openrouters"):
            self.client = OpenAI(
                base_url=os.environ.get('OPENROUTER_ENDPOINT'),
                api_key=os.environ.get('OPENROUTER_API_KEY'),
            )
        elif self.backend == "publicapi":
            self.client = OpenAI(
                base_url=os.environ.get('OPENROUTER_ENDPOINT'),
                api_key=os.environ.get('OPENROUTER_API_KEY'),
                default_headers={
                    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    "User-Agent": "MyPythonClient/1.0"
                }
            )
        else:
            raise NotImplementedError
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        It guarantees the cleanup of the model.
        """
        print(f"Cleaning up model '{self.model_id}'...")
        if self.model is not None:
            del self.model

        # Force garbage collection and clear PyTorch's memory cache
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate_text(self, prompts: list[str], batch_size: int = 4,
                        tokenizer_args=None,
                      generator_args=None,
                      include_thinking=False,
                      generated_only=False):
        if self.backend == "huggingface":
            return  self.generate_text_hf(prompts, tokenizer_args, generator_args, generated_only)
        elif self.backend == "ollama":
            return self.generate_text_ollama(prompts, tokenizer_args, generator_args, include_thinking)
        elif (self.backend == "openrouters") | (self.backend == "publicapi"):
            return self.generate_text_openrouter(prompts, tokenizer_args, generator_args)

    def generate_text_cot(self, prompts: list[str], batch_size: int = 4,
                        tokenizer_args=None,
                      generator_args=None,
                      include_thinking=False,
                      generated_only=False):
        if self.backend == "huggingface":
            return self.generate_text_cot_hf(prompts, tokenizer_args, generator_args)
        elif self.backend == "ollama":
            raise NotImplementedError
        elif self.backend == "openrouters":
            return self.generate_text_cot_openrouter(prompts, tokenizer_args, generator_args)

    def generate_text_ollama(self, prompt: str,
                             tokenizer_args=None, generator_args=None, include_thinking=False) -> str:
        resp = self.model._generate(prompts=[prompt])
        if include_thinking and ('thinking' in resp.generations[0][0].generation_info):
            output = resp.generations[0][0].generation_info['thinking'] + '\n' + resp.generations[0][0].text
        else:
            output = resp.generations[0][0].text
        return output

    def generate_text_hf(self, prompt: str, tokenizer_args=None, generator_args=None,
                         generated_only=False):
        inputs = self.tokenizer(prompt, **tokenizer_args).to(self.model.device)
        if generated_only:
            input_length = inputs.input_ids.shape[1]
        else:
            input_length = 0
        outputs = self.model.generate(**inputs, **generator_args)
        texts = self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        return texts

    def generate_text_cot_hf(self, prompts: list[str], tokenizer_args=None, generator_args=None,generated_only=False):
        messages = []
        for q in prompts:
            messages.append({"role": "user", "content": q})
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generator_args
                )

            prompt_len = model_inputs["input_ids"].shape[1]
            new_tokens = output_tokens[0][prompt_len:]

            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": answer})
        return messages

    def generate_text_cot_openrouter(self, prompts: list[str], tokenizer_args=None, generator_args=None,
                         generated_only=False):
        if 'max_new_tokens' in generator_args:
            generator_args['max_tokens'] = generator_args['max_new_tokens']
            generator_args.pop('max_new_tokens')
        current_time = time.time()
        random.seed(current_time)
        i=random.randint(1, 10000)
        messages = [{"role": "system", "content": f"Request ID: {i}"}]

        for q in prompts:
            messages.append({"role": "user", "content": q})

            response = self.client.chat.completions.create(
                model=self.model_id,
                extra_body={"reasoning": {"enabled": False},
                            "include_thoughts": False},
                messages=messages,
                **generator_args
            )

            answer = response.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})
        return messages

    def generate_text_openrouter(self, prompt: str, tokenizer_args=None, generator_args=None,
                         generated_only=False):
        if 'max_new_tokens' in generator_args:
            generator_args['max_tokens'] = generator_args['max_new_tokens']
            generator_args.pop('max_new_tokens')
            generator_args.pop('do_sample')
            generator_args.pop('top_k')
        current_time = time.time()
        random.seed(current_time)
        i=random.randint(1, 10000)
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": f"Request ID: {i}"},
                {"role": "user", "content": prompt}
            ],
            # OpenRouter uses 'extra_body' to pass provider-specific settings
            extra_body={
        "thinking_config": {
            "include_thoughts": False
        }
    },
            reasoning_effort='low',
            **generator_args
        )
        if completion.choices:
            return completion.choices[0].message.content
        else:
            return None

    def generate_logits_hf(self, prompts: list[str], batch_size: int = 4,
                           tokenizer_args=None, generator_args=None):
        # Set default arguments if not provided
        if generator_args is None:
            generator_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}

        # Initialize list to store all outputs
        all_sequences = []
        all_scores = []
        all_logits = []
        max_input_length = 0

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            # Slice the current batch of prompts
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize the current batch
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", **tokenizer_args).to(self.model.device)

            # Track the maximum input length
            max_input_length = max(max_input_length, inputs['input_ids'].shape[1])

            # Generate for the current batch
            with torch.no_grad():
                batch_outputs = self.model.generate(**inputs, **generator_args)

            # Collect batch outputs
            all_sequences.append(batch_outputs.sequences)

            # Collect additional attributes if they exist
            if hasattr(batch_outputs, 'scores') and batch_outputs.scores is not None:
                all_scores.append(batch_outputs.scores)

            if hasattr(batch_outputs, 'logits') and batch_outputs.logits is not None:
                all_logits.append(batch_outputs.logits)

        final_outputs = type(batch_outputs)(
            sequences=torch.cat(all_sequences, dim=0) if all_sequences else None,
            scores=tuple(torch.cat(logit_tuple, dim=0) for logit_tuple in zip(*all_logits)) if all_scores else None,
            logits=tuple(torch.cat(logit_tuple, dim=0) for logit_tuple in zip(*all_logits)) if all_logits else None
        )
        return max_input_length, final_outputs