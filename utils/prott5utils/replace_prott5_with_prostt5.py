import pandas as pd
import numpy as np
from tqdm import tqdm

def align_embeddings(old_csv, new_csv, output_csv):
    """
    Align embeddings from new CSV to match old CSV's structure
    """
    # Read the old CSV without headers
    print("Reading files...")
    old_df = pd.read_csv(old_csv, header=None)
    # First two columns are uniprot_id and position
    old_df.columns = ['uniprot_id', 'position'] + [i for i in range(1024)]
    
    # Read new CSV with headers (since it has them)
    new_df = pd.read_csv(new_csv)
    
    # Create lookup dictionary from new embeddings
    print("Creating lookup dictionary from new embeddings...")
    new_embeddings = {}
    for _, row in new_df.iterrows():
        key = (row['uniprot_id'], row['position'])
        # Get all columns except uniprot_id and position
        embedding = row.iloc[2:].values
        new_embeddings[key] = embedding
    
    # Process and save without headers
    print("Creating and saving aligned embeddings...")
    with open(output_csv, 'w') as f:
        for _, row in tqdm(old_df.iterrows(), total=len(old_df), desc="Processing rows"):
            key = (row['uniprot_id'], row['position'])
            if key in new_embeddings:
                # Get new embedding
                embedding = new_embeddings[key]
                # Format line: uniprot_id,position,v1,v2,...,v1024
                line = f"{row['uniprot_id']},{row['position']}," + \
                       ','.join(f'{x:.8f}' for x in embedding)
                f.write(f"{line}\n")
            else:
                print(f"Warning: Could not find embedding for {key}")
    
    print("Done!")

# Usage
old_csv = "LMSuccSite/data/train/features/train_positive_ProtT5-XL-UniRef50.csv"
new_csv = "LMSuccSite/data/grouped/embeddings_prost_train.csv"
output_csv = "LMSuccSite/data/train/features/train_positive_ProstT5.csv"

align_embeddings(old_csv, new_csv, output_csv)