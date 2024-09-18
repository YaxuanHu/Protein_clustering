import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.pretrained import ESM3_sm_open_v0
import pytorch_lightning as pl

# Define max sequence length
MAX_LENGTH = 256

# Custom dataset class for loading sequences from CSV
class SequenceDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=MAX_LENGTH):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Sequence']
        seq_id = self.data.iloc[idx]['ID']
        encoded_sequence = self.tokenizer.encode(sequence, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return seq_id, encoded_sequence[0]

# Custom PyTorch Lightning Module for ESM3
class ESM3LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ESM3_sm_open_v0("cuda")
        
    def forward(self, sequence_tokens):
        with torch.no_grad():
            model_embedding = self.model(sequence_tokens=sequence_tokens.cuda())
        return model_embedding

    def predict_step(self, batch, batch_idx):
        seq_ids, sequences = batch
        output = self(sequence_tokens=sequences.cuda())
        return seq_ids, output

# Function to save embeddings with corresponding IDs
def save_embeddings(seq_ids, embeddings, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for seq_id, embedding in zip(seq_ids, embeddings.sequence_logits):
        torch.save(embedding.cpu(), os.path.join(save_dir, f"{seq_id}.pt"))

# Function to process all CSV files in a directory
def process_all_csv_files(csv_directory, ebd_directory, gpus):
    tokenizer = EsmSequenceTokenizer()  # Initialize the tokenizer
    
    # Get a list of all CSV files in the root directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        save_subdir = os.path.join(ebd_directory, os.path.splitext(csv_file)[0])  # Subfolder named after the CSV file
        
        # Initialize the dataset and dataloader
        dataset = SequenceDataset(csv_path, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        # Initialize the model
        model = ESM3LightningModule()
        
        # Create a PyTorch Lightning Trainer with specific GPU settings
        trainer = pl.Trainer(
            accelerator="gpu",  # Use GPU
            devices=gpus,  # List of GPU IDs to use
            strategy="ddp",  # Distributed Data Parallel
            precision=16,  # Mixed precision
            max_epochs=1
        )
        
        # Predict and save embeddings
        predictions = trainer.predict(model, dataloader)
        
        # Collect embeddings and save
        for batch in predictions:
            seq_ids, embeddings = batch
            save_embeddings(seq_ids, embeddings, save_subdir)

# Example usage
csv_directory = './data/csv/'     # Replace with your desired CSV output path
ebd_directory = './data/ebd/'
gpus = [0, 1]  # List of GPU IDs to use, for example GPU 0 and GPU 1

process_all_csv_files(csv_directory, ebd_directory, gpus)
