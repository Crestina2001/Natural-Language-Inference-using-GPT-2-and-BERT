from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Check if CUDA is available and select device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

def analyze_relationship(sentence_a, sentence_b):
    """
    Function to prompt GPT-2 to analyze the relationship between two sentences.
    """
    prompt = (f"Determine if the relationship between the following sentences is "
              f"contradictory, entailment, or neutral:\n"
              f"Sentence A: '{sentence_a}'\nSentence B: '{sentence_b}'\n"
              f"Answer:")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=60, num_return_sequences=1, early_stopping=True)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Answer:")[1].strip()

# Example sentences
sentence_pairs = [
    ("The cat is sleeping on the mat", "The cat is awake"),
    ("John is a bachelor", "John is married"),
    # Add more pairs as needed
]

# Analyzing each pair
for sentence_a, sentence_b in sentence_pairs:
    analysis = analyze_relationship(sentence_a, sentence_b)
    print(f"Sentence A: '{sentence_a}'\nSentence B: '{sentence_b}'\nAnalysis: {analysis}\n")
