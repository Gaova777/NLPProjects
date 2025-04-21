import pandas as pd
import re
import torch
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc

def remove_parenthesis(text):
    return re.sub(r'\(.*?\)', '', text)

class CharacterChatBot:
    def __init__(self,
                 model_path,
                 data_path='data/naruto.csv',
                 huggingface_token=None):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = 'meta-llama/Meta-Llama-3-8B-instruct'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print('Model not found in Hugging Face hub, training a new one...')
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model = self.load_model(self.model_path)

    def chat(self, message, history=[]):
        messages = []
        messages.append({"role": "system", "content": """You are Naruto Uzumaki, a ninja from the Hidden Leaf Village. You are known for your determination and never-give-up attitude. Your responses should reflect the personality and speech patterns.\n"""})
        
        for msg in history:
            if isinstance(msg, dict) and "user" in msg and "naruto" in msg:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["naruto"]})
        
        messages.append({"role": "user", "content": message})

        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"[SYSTEM]: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"[USER]: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"[NARUTO]: {msg['content']}\n"
        prompt += "[NARUTO]:"

        output = self.model(
            prompt,
            max_length=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False
        )

        return output[0]['generated_text']

    def load_model(self, model_path):
        pipe = transformers.pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=-1  # Forzar uso de CPU
        )
        return pipe

    def train(self,
        base_model_name_or_path,
        dataset,
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=300,
        warmup_ratio=0.3,
        lr_scheduler_type="constant"):

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # SFTConfig SIN peft_config
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none",
            fp16=False,
            max_seq_length=512,
        )

        # Entrenador con peft_config en el lugar correcto
        trainer = SFTTrainer(
           model=model,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config
        )

        trainer.train()

        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float32
        )

        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()


    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        df['line'] = df["line"].apply(remove_parenthesis)
        df['number_of_words'] = df["line"].str.strip().str.split(" ").apply(len)
        df['naruto_response_flag'] = 0
        df.loc[(df['name'] == "Naruto") & (df["number_of_words"] > 5), 'naruto_response_flag'] = 1

        indices = df[(df["naruto_response_flag"] == 1) & (df.index > 0)].index
        prompts = []
        system_prompt = '''You are Naruto from the anime "Naruto". Your responses should reflect the personality and speech patterns of Naruto.\n'''

        for idx in indices:
            prompt = system_prompt
            prompt += df.iloc[idx - 1]['line'] + "\n"
            prompt += df.iloc[idx]['line']
            prompts.append(prompt)

        return Dataset.from_pandas(pd.DataFrame({"text": prompts}))
