# Run this script to extract distance maps from PDB files and add them to the structure feature dataset.

import numpy as np
from Bio import PDB
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pandas as pd

def create_distance_map(pdb_file, center_pos, window_size=16):
    """
    Create distance map from PDB file with proper edge case handling
    Args:
        pdb_file: Path to PDB file
        center_pos: Position of central residue (K)
        window_size: Window size (default 16 for 33 total)
    Returns:
        distance_map: 33x33 distance map with proper padding (infinity for no contact)
    """
    try:
        # Parse PDB
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        # Calculate window boundaries
        total_length = 2 * window_size + 1  # 33
        start_pos = center_pos - window_size
        end_pos = center_pos + window_size + 1
        
        # Initialize distance map with infinity (for padding/no contact)
        distance_map = np.full((total_length, total_length), np.inf)
        
        # Get list of valid residue positions in PDB
        valid_residues = {}
        for residue in model.get_residues():
            res_id = residue.id[1]
            try:
                valid_residues[res_id] = residue['CA'].get_coord()
            except KeyError:
                continue
        
        # Calculate needed padding
        left_pad = max(0, -(start_pos - 1))
        
        # For each position in our window
        for i in range(total_length):
            pos_i = start_pos + i
            
            # Skip if position is not in structure
            if pos_i not in valid_residues:
                continue
                
            coord_i = valid_residues[pos_i]
            
            for j in range(total_length):
                pos_j = start_pos + j
                
                # Skip if position is not in structure
                if pos_j not in valid_residues:
                    continue
                    
                coord_j = valid_residues[pos_j]
                
                # Calculate distance
                dist = np.linalg.norm(coord_i - coord_j)
                distance_map[i, j] = dist
        
        return distance_map
        
    except Exception as e:
        print(f"Error creating distance map: {e}")
        return np.full((2 * window_size + 1, 2 * window_size + 1), np.inf)

def add_distance_maps_to_data(input_file, output_file, pdb_dir):
    """Add distance maps to existing data"""
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize new column for distance maps
    distance_maps = []
    
    print("Generating distance maps...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Count padding needed (like in sequence)
            n_dashes = len(row['sequence']) - len(row['sequence'].lstrip('-'))
            
            # Create distance map
            pdb_file = f"{pdb_dir}/AF-{row['entry']}.pdb"
            distance_map = create_distance_map(pdb_file, row['pos'])
            
            # Replace infinity with -1 for better storage
            distance_map[np.isinf(distance_map)] = -1
            
            # Flatten and convert to string format
            distance_map_str = ','.join(map(str, distance_map.flatten()))
            distance_maps.append(distance_map_str)
            
            # Debug first few rows
            if idx < 3:
                print(f"\nRow {idx + 1}:")
                print(f"Entry: {row['entry']}")
                print(f"Position: {row['pos']}")
                print(f"Padding (dashes): {n_dashes}")
                print(f"Distance map shape: {distance_map.shape}")
                print(f"Distance range: {np.min(distance_map[distance_map != -1]):.2f} to {np.max(distance_map[distance_map != -1]):.2f}")
                print(f"First few values: {distance_map_str[:50]}...")
                
        except Exception as e:
            print(f"Error processing entry {row['entry']}: {e}")
            # Add distance map with -1 in case of error
            distance_map = np.full((33, 33), -1)
            distance_maps.append(','.join(map(str, distance_map.flatten())))
    
    # Add distance maps to DataFrame
    df['distance_map'] = distance_maps
    
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

def visualize_distance_map(distance_map_str, sequence, threshold=None):
    """
    Visualize a distance map with optional threshold highlighting
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert string back to 33x33 matrix
    values = np.array([float(x) for x in distance_map_str.split(',')])
    distance_map = values.reshape(33, 33)
    
    # Create mask for invalid/padding distances
    mask = distance_map == -1
    
    plt.figure(figsize=(12, 5))
    
    # Plot raw distances
    plt.subplot(1, 2, 1)
    sns.heatmap(distance_map, mask=mask, cmap='viridis_r', 
                vmin=0, vmax=20, cbar_kws={'label': 'Distance (Å)'})
    plt.title('Raw Distances')
    plt.xlabel('Residue Position')
    plt.ylabel('Residue Position')
    
    # If threshold provided, also plot contact map
    if threshold is not None:
        plt.subplot(1, 2, 2)
        contact_map = (distance_map < threshold) & ~mask
        sns.heatmap(contact_map, cmap='binary', 
                    cbar_kws={'label': f'Contact (< {threshold}Å)'})
        plt.title(f'Contact Map (threshold: {threshold}Å)')
        plt.xlabel('Residue Position')
        plt.ylabel('Residue Position')
    
    plt.suptitle(f'Distance Map Analysis\nSequence: {sequence}')
    plt.tight_layout()
    plt.show()

# This will add the distance maps to the existing dataset
# and save it to a new CSV file.
# Make sure to adjust the input and output file paths as needed.
# Remember to run the 1_extract_features.py script first to generate the initial dataset.

if __name__ == "__main__":
    input_file = "training_code/data/processed_features_train.csv"
    output_file = "training_code/data/processed_features_train_contactmap.csv"
    pdb_dir = "data/structure/train"
    add_distance_maps_to_data(input_file, output_file, pdb_dir)