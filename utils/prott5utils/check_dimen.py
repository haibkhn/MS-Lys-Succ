import pandas as pd
import numpy as np

def check_embedding_dimension(csv_file):
    """
    Check embedding dimension from a CSV without headers
    """
    # Read the first line to count columns
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
        num_columns = len(first_line.split(','))
    
    print(f"File: {csv_file}")
    print(f"Total columns: {num_columns}")
    print(f"Embedding dimension: {num_columns - 2}")  # Subtract 2 for uniprot_id and position columns
    
    # Read first few rows to verify
    df = pd.read_csv(csv_file, header=None, nrows=1)
    print(f"Shape of first row: {df.shape}")
    print("--------------------")

# Check each file
check_embedding_dimension("embeddings_prost_train.csv")
check_embedding_dimension("embeddings_prost_test.csv")
check_embedding_dimension("embeddings_esm2.csv")