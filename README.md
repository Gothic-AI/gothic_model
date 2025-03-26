# Gothic AI - Chatbot for Novel Character

This project aims to build a chatbot that emulates a character from a novel using a fine-tuned GPT-2 model. The chatbot will generate responses based on the character's personality as depicted in the novel.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)

## Installation

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Gothic-AI/gothic_model.git
cd gothic_model
```

### Step 2: Set up a virtual environment
Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install the required dependencies
```
pip install -r requirements.txt
```

# Model Setup Instructions

## Prepare Your Training Data

Before running the model, you need to prepare the environment by providing the necessary data files. Your training data should be organized as follows:

1. **Plain Text Version of the Novel**:  
   Place the plain text version of the novel in the following path:

data/test_plain.txt

2. **Q&A Version of the Novel**:  
Place the Q&A version of the novel in the following path:

data/test_q&a.txt

## Configuration

You need to modify the configuration file based on your experiment setup.

1. If you are using the **Plain Text** version of the novel, modify the `config/config_plain.yaml` file.

2. If you are using the **Q&A** version of the novel, modify the `config/config_q&a.yaml` file.

In the configuration file, you can adjust various preferences such as training parameters, learning rates, and other model settings according to your specific needs.

---

Once your data and configuration are ready, you can proceed with running the model!


Training the Model

To train the GPT-2 model on your dataset, follow these steps:

Prepare the data:
Make sure your data is in the data/ folder and is in .txt format (e.g., test_plain.txt, test_q&a.txt).
Run the training script:
To train the model on plain text data, run:

python scripts/train_model.py --config config/config_plain.yaml
To train the model on Q&A data, run:

python scripts/train_model.py --config config/config_q&a.yaml
This command will start the fine-tuning process based on the configuration specified in the selected YAML file.

The training process may take a while depending on your hardware and the size of your data.
Model Output:
Once the training completes, the model will be saved in the models/ directory (as specified in the respective config/*.yaml file).
Generating Text

Once the model is trained, you can use it to generate text (simulating responses from the Gothic chatbot).

Run the text generation script:
To generate text, use the following command:

python inference/generate_text.py
You can modify the generate_text.py script to provide different prompts for your chatbot, or change the model and output sequence length as required.
Modify the prompt:
In inference/generate_text.py, you can modify the sequence1 variable to set different questions or inputs for the chatbot.
Folder Structure

Here's a brief overview of the folder structure:

gothic_model/
├── config/
│   ├── config_plain.yaml        # Configuration file for training with plain text data
│   └── config_q&a.yaml          # Configuration file for training with Q&A data
├── data/
│   ├── test_plain.txt           # Your training data (plain text version of the novel)
│   └── test_q&a.txt             # Your training data (Q&A version of the novel)
├── inference/
│   ├── generate_text.py         # Script to generate text based on the trained model
│   └── utils.py                 # Helper functions for loading model and tokenizer
├── models/                      # Directory where the trained models are saved
├── scripts/
│   ├── data.py                  # Data handling and processing
│   └── train_model.py           # Main script for training the model
├── readme.md                    # This readme file
├── requirements.txt             # List of dependencies
└── .gitignore                   # Ignore unnecessary files in Git
Troubleshooting

If you encounter any issues, here are a few common troubleshooting tips:

File Not Found Errors:
Make sure all files referenced in the config/*.yaml files exist in the proper directories. Verify the paths and filenames.
Memory Issues:
If your model is too large or your system runs out of memory, try reducing the batch size or number of epochs in the configuration file.
Tokenization Issues:
If you encounter issues with tokenization, ensure your data is properly formatted and free of any non-text elements that could interfere with the tokenizer.
CUDA/GPUs:
If you're using a GPU and facing issues related to CUDA, make sure that your system has the correct CUDA version installed. Ensure the proper GPU drivers are configured.