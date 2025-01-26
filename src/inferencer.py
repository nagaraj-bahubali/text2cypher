import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from dotenv import load_dotenv

class Inferencer:
    def __init__(self, base_model, fine_tuned_model, config_path):
        
        # Load configuration parameters
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['inference']
        
        self.base_model = base_model 
        self.fine_tuned_model = fine_tuned_model 

    def get_pipeline(self):
        # Load base model 
        load_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            **self.config['model']
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Merge base model with LoRA weights
        model = PeftModel.from_pretrained(load_model, self.fine_tuned_model)
        model = model.merge_and_unload()

        # Initialize the text generation pipeline fine-tuned model and tokenizer
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            **self.config['pipeline']
        )

        return text_generation_pipeline
