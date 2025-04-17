import argparse
import os
import yaml
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

# Import dataset and data collator functions from data.py
from data import load_dataset, load_data_collator
from evalution_metrics import compute_metrics

def train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Load the dataset and data collator
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
      
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Save the model
    model.save_pretrained(output_dir)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()

def main():
    # Set up argparse to take a config file as input
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file (can omit .yaml extension)')
    args = parser.parse_args()

    # Check if the file path ends with '.yaml', if not, append it
    config_file = args.config
    if not config_file.endswith('.yaml'):
        config_file += '.yaml'

    # Load configuration from the provided YAML file
    if not os.path.exists(config_file):
        print(f"Error: {config_file} does not exist.")
        return

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extract values from the config
    train_file_path = config['train_file_path']
    model_name = config['model_name']
    output_dir = config['output_dir']
    overwrite_output_dir = config['overwrite_output_dir']
    per_device_train_batch_size = config['per_device_train_batch_size']
    num_train_epochs = config['num_train_epochs']
    save_steps = config['save_steps']

    # Train the model using the loaded configuration
    train(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

if __name__ == "__main__":
    main()