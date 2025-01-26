from src.finetuner import FineTuner

# Path to the config file
config_path = 'config/exp_1.yaml'

# Initialize the FineTuner
finetuner = FineTuner(config_path)

# Setup the trainer
trainer = finetuner.setup_trainer()

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
fine_tuned_model_path = finetuner.config['fine_tuned_model']
trainer.model.save_pretrained(fine_tuned_model_path)
trainer.tokenizer.save_pretrained(fine_tuned_model_path)
