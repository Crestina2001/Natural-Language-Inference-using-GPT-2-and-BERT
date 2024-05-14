import torch
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.to(device)  # Move the model to the GPU
model.eval()

# Function to calculate log-likelihood
def log_likelihood(sentence, tokenizer, model, device):
    tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
    return -outputs.loss.item()

# Load and preprocess CrowS-Pairs dataset
dataset_path = 'dataset/crows_pairs_anonymized.csv'
df = pd.read_csv(dataset_path)

# Filter for age-related sentence pairs
age_df = df[df['bias_type'] == 'age'].sample(3)  # Sample 3 examples for the case study

# Prepare sentences and calculate scores
case_study_results = []
for _, row in age_df.iterrows():
    stereotype_sentence = row['sent_more']
    astereotype_sentence = row['sent_less']
    ll_stereotype = log_likelihood(stereotype_sentence, tokenizer, model, device)
    ll_astereotype = log_likelihood(astereotype_sentence, tokenizer, model, device)
    case_study_results.append({
        "Sentence (Stereotypical)": stereotype_sentence,
        "Sentence (Astereotypical)": astereotype_sentence,
        "Log-Likelihood Score (Stereo)": ll_stereotype,
        "Log-Likelihood Score (Astereo)": ll_astereotype
    })

# Convert results to DataFrame for display
case_study_df = pd.DataFrame(case_study_results)
print(case_study_df.to_markdown(index=False))

# Calculate pseudo-log-likelihood for each pair in the entire dataset
results = []
for stereotype, astereotype in sentence_pairs:
    ll_stereotype = log_likelihood(stereotype, tokenizer, model, device)
    ll_astereotype = log_likelihood(astereotype, tokenizer, model, device)
    results.append((ll_stereotype, ll_astereotype))

# Analyze the results
biased_count = sum(1 for pll_s, pll_a in results if pll_s > pll_a)
total_count = len(results)
bias_percentage = biased_count / total_count * 100

print(f"Bias Percentage: {bias_percentage}% (Ideal: 50%)")
