from utils import generate_text

model1_path = "./models/plain_text_model/"
model2_path = "./models/q&a_text_model/"
sequence1 = "Q: Why does Elara ask Jareth about the sky?"
max_len = 50

# generated_text = generate_text(model1_path, sequence1, max_len)
generated_text = generate_text(model2_path, sequence1, max_len)
print(generated_text)  # This can be printed or used further

