import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
from tqdm import tqdm
from bert_dataloader import load_jsonl, prepare_data_for_model, split_data, NLIDataset
from bert_training import train_epoch, evaluate, test_accuracy
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import argparse
import json
import datetime



if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()

    # Extract arguments
    batch_size = args.bs
    learning_rate = args.lr
    num_epochs = 5
    max_length = 512

    device = 'cuda:1'
    # Load and prepare data
    filename = 'dataset/mismatched.jsonl'
    data = load_jsonl(filename)
    model_inputs, labels = prepare_data_for_model(data)
    
    # Split data into train, val, and test sets
    inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test = split_data(
        model_inputs, labels, train_size=0.8, test_size=0.2)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    model.to(device)  # Move model to GPU if available
    
    # Assuming tokenizer and max_length are already defined
    train_dataset = NLIDataset(inputs_train, labels_train, tokenizer, max_length)
    val_dataset = NLIDataset(inputs_val, labels_val, tokenizer, max_length)
    test_dataset = NLIDataset(inputs_test, labels_test, tokenizer, max_length)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")


        # Check if this is the best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            print(f"New best model found at epoch {epoch+1}")
            logging.info(f"New best model found at epoch {epoch+1}")
    # At the end of training, reload the best model state
    model.load_state_dict(best_model_state)
    
    test_acc = test_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.3f}")
    logging.info(f"Test Accuracy: {test_acc:.3f}")
    # Prepare results data
    results = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'best_val_loss': best_val_loss,
        'test_accuracy': test_acc
    }


    # Create a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f'results/results_{timestamp}.json'

    # Write results to a JSON file
    with open(results_filename, 'w') as f:
        json.dump(results, f)



