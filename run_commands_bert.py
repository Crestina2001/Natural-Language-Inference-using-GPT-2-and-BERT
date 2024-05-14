import subprocess

# Define different configurations
batch_sizes = [16, 32]
learning_rates = [2e-5, 3e-5, 5e-5]

# Run training for each configuration
for bs in batch_sizes:
    for lr in learning_rates:
                subprocess.run([
                    'python', 'bert_main.py', 
                    '--bs', str(bs), 
                    '--lr', str(lr)
                ])