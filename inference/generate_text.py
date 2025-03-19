from utils import generate_text

model1_path = "./models/plain_text_model/"
sequence1 = "[Q] What is the novel name?"
max_len = 50
generate_text(model1_path, sequence1, max_len) 