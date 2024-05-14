# Natural Language Inference using GPT-2 and BERT

This project explores the implementation and fine-tuning of GPT-2 and BERT models for Natural Language Inference (NLI) tasks using the MultiNLI dataset. The project also includes an analysis of age-related biases using the CrowS-Pairs dataset.

## Table of Contents
- [Overview](#overview)
- [Experiment Details](#experiment-details)
  - [Task 1: Natural Language Inference](#task-1-natural-language-inference)
  - [Task 2: Bias Analysis](#task-2-bias-analysis)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This project focuses on two main tasks:
1. **Natural Language Inference (NLI)**: Implementing and fine-tuning GPT-2 and BERT models on the MultiNLI dataset to classify sentence pairs into categories of entailment, contradiction, or neutrality.
2. **Bias Analysis**: Evaluating age-related biases in the BERT model using the CrowS-Pairs dataset.

## Experiment Details

### Task 1: Natural Language Inference

#### 1.1 Prompting

**Experiment Overview**: 
- Implemented a prompting approach using GPT-2 for NLI tasks.
- Prompt format: 
  ```python
  prompt = (f"Determine if the relationship between the following sentences is "
            f"contradictory, entailment, or neutral:\n"
            f"Sentence A: ’{sentence_a}’\nSentence B: ’{sentence_b}’\n"
            f"Answer:")
  ```

**Results**: 
- GPT-2’s responses often did not align with the expected categories.

**Analysis**:
- The unsatisfactory performance can be attributed to the complexity of NLI tasks, GPT-2’s limitations in understanding context, and the potential inadequacy of the prompt structure.

#### 1.2 Fine-Tuning

**Dataset**:
- Utilized a pre-selected subset of the MultiNLI dataset, divided into 'matched' and 'mismatched' sets.

**Model Setup**:
- Used GPT-2 and BERT models with specific configurations and hyperparameters.

**Results**:
- **GPT-2**: Best test accuracy on the mismatched dataset was 83.87%.
- **BERT**: Best test accuracy on the mismatched dataset was 72.98%.

**Analysis**:
- GPT-2 showed better performance than BERT in NLI tasks.

### Task 2: Bias Analysis

#### 2.1 Measurement Setup

**Social Domain**:
- Focused on age-related biases using the CrowS-Pairs dataset.

**Metric**:
- Utilized the pseudo-log-likelihood score to measure the likelihood of stereotypical versus astereotypical sentences.

**Implementation Details**:
- Used the pre-trained `bert-base-uncased` model without additional fine-tuning.

#### 2.2 Experiment Results

**Quantitative Results**:
- Bias percentage was approximately 45.98%.

**Case Study**:
- Demonstrated bias in specific sentence pairs related to age.

**Conclusion**:
- The BERT model generally demonstrated a balanced approach to age-related content but showed slight deviations from an ideal unbiased score.

## Results

The project yielded several insights into the performance and biases of GPT-2 and BERT models in NLI tasks. Detailed results and analyses are documented in the report.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the main script, use the following command:

```bash
python main.py --bs <batch_size> --lr <learning_rate> --non_lin <non_linearity> --red_factor <reduction_factor>
```

For example:

```bash
python main.py --bs 32 --lr 1e-4 --non_lin swish --red_factor 16
```

## License

This project is licensed under the MIT License.
