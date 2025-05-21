import torch
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import gc
import os

def batch_encode_sequences(model, tokenizer, sequences, batch_size=32, device='cpu'):
    """
    Encode sequences in batches for ProstT5
    """
    all_embeddings = []
    
    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        
        # Add spaces between amino acids and handle special characters
        batch_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch_sequences]
        
        # Add prefix for amino acid sequences
        batch_sequences = ["<AA2fold> " + seq for seq in batch_sequences]
        
        # Tokenize with PyTorch tensors
        ids = tokenizer.batch_encode_plus(
            batch_sequences, 
            add_special_tokens=True, 
            padding="longest",
            return_tensors='pt'
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            embedding = model(
                input_ids=ids.input_ids,
                attention_mask=ids.attention_mask
            )
            embedding = embedding.last_hidden_state.cpu().numpy()
        
        # Extract embeddings for each sequence (skip the prefix token)
        for seq_num in range(len(embedding)):
            seq_len = (ids.attention_mask[seq_num] == 1).sum()
            seq_emb = embedding[seq_num][1:seq_len-1]  # Skip prefix and end tokens
            all_embeddings.append(seq_emb)
            
        # Clean up GPU memory
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    return all_embeddings

def process_sequences_and_save(csv_file, output_file, temp_dir='temp_embeddings', batch_size=32):
    """
    Process sequences from CSV and save embeddings with checkpoint capability
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load model and tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", 
                                          do_lower_case=False, cache_dir="/home/ubuntu/data/hai/huggingface_cache")
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", cache_dir="/home/ubuntu/data/hai/huggingface_cache").to(device)
    
    # Set precision based on device
    if device == 'cpu':
        model.full()
    else:
        model.half()
    
    model = model.eval()
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Check which proteins have already been processed
    processed_proteins = set()
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file)
        processed_proteins = set(processed_df['uniprot_id'].unique())
        print(f"Found {len(processed_proteins)} already processed proteins")
    
    # Write header if file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write(f"uniprot_id,position," + ','.join(f'v{i+1}' for i in range(1024)) + '\n')
    
    # Process each unique protein
    for _, row in tqdm(df.iterrows(), desc="Processing proteins", total=len(df)):
        uniprot_id = row['uniprot_id']
        
        # Skip if already processed
        if uniprot_id in processed_proteins:
            print(f"Skipping {uniprot_id} - already processed")
            continue
            
        sequence = row['full_sequence']
        
        try:
            # Handle positions - fixed version
            if isinstance(row['positions'], str):
                positions = [int(pos.strip()) for pos in row['positions'].strip('"').split(',')]
            else:
                positions = [int(row['positions'])]
            
            # Get embeddings (only once per sequence)
            embeddings = batch_encode_sequences(model, tokenizer, [sequence], 
                                             batch_size=1, device=device)[0]
            
            # Save embeddings for this protein to temporary file
            temp_file = os.path.join(temp_dir, f"{uniprot_id}_temp.csv")
            with open(temp_file, 'w') as f:
                for pos in positions:
                    array_pos = pos - 1
                    embedding = embeddings[array_pos]
                    embedding_str = ','.join(f'{x:.8f}' for x in embedding)
                    f.write(f"{uniprot_id},{pos},{embedding_str}\n")
            
            # Append temp file to main output file
            with open(temp_file, 'r') as temp, open(output_file, 'a') as out:
                out.write(temp.read())
            
            # Remove temp file and mark as processed
            os.remove(temp_file)
            processed_proteins.add(uniprot_id)
            
        except Exception as e:
            print(f"Error processing {uniprot_id}: {str(e)}")
            continue
            
        # Clean up GPU memory periodically
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    # Clean up
    del model
    del tokenizer
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        print(f"Note: {temp_dir} not empty, some temporary files may remain")

# Usage
csv_file = "grouped/test.csv"  # Your CSV file with columns: uniprot_id, positions, full_sequence
output_file = "embeddings_prost_test.csv"
temp_dir = "temp_embeddings"

# Run processing
process_sequences_and_save(csv_file, output_file, temp_dir, batch_size=32)