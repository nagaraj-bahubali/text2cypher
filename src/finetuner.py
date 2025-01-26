import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

class FineTuner:
    def __init__(self, config_path):
        # Load configuration parameters
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['fine_tuning']

        self.base_model = self.config['base_model']
        self.device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        print(f"Model loaded on {self.device}")

        # Set quantization parameters
        self.quant_config = BitsAndBytesConfig(**self.config['quant_config'], bnb_4bit_compute_dtype=torch.bfloat16)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.quant_config,
            **self.config['model']
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load LoRA configuration
        self.peft_args = LoraConfig(**self.config['peft_args'])

        # Load training parameters
        self.training_params = TrainingArguments(**self.config['training_params'])

    def setup_trainer(self):
        dataset = load_dataset('parquet', data_files=self.config['dataset_path'])
        self.trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_args,
            dataset_text_field='text',
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=False,
            train_dataset=dataset["train"])
        return self.trainer
