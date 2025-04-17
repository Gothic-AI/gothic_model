import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# def generate_text(model_path, sequence, max_length):
    
#     model = load_model(model_path)
#     tokenizer = load_tokenizer(model_path)
#     ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
#     final_outputs = model.generate(
#         ids,
#         do_sample=True,
#         max_length=max_length,
#         pad_token_id=model.config.eos_token_id,
#         top_k=50,
#         top_p=0.95,
#     )
#     print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))



# def generate_text(model_path, sequence, max_length, temperature=1.0, top_k=50, top_p=0.95):
#     model = load_model(model_path)
#     tokenizer = load_tokenizer(model_path)
    
#     # Encode input text
#     ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create attention mask: 1 for real tokens, 0 for padding tokens
#     attention_mask = torch.ones(ids.shape, device=ids.device)  # Default attention mask (all ones)
    
#     # Generate text
#     final_outputs = model.generate(
#         ids,
#         attention_mask=attention_mask,  # Explicitly pass the attention mask
#         do_sample=True,
#         max_length=max_length,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p,
#         pad_token_id=model.config.eos_token_id,  # EOS token used as pad token
#     )
    
#     return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


def generate_text(model_path, sequence, max_length, temperature=1.0, top_k=50, top_p=0.95):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 💥 Move model to the same device as input

    # Encode input text
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to(device)

    # Create attention mask
    attention_mask = torch.ones(ids.shape, device=device)

    # Generate text
    final_outputs = model.generate(
        ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=model.config.eos_token_id,
    )

    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)
