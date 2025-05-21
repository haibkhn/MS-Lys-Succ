import numpy as np
import pandas as pd
import pyrosetta
from Bio import SeqIO
from tqdm import tqdm
import os
import csv
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_pyrosetta():
    """Initialize PyRosetta with options"""
    try:
        pyrosetta.init('-mute all')
        logging.info("PyRosetta initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize PyRosetta: {e}")
        raise

def parse_fasta_header(header):
    """Parse FASTA header to get UniProt ID and position"""
    try:
        parts = header.split('|')
        if len(parts) >= 2:
            uniprot_id = parts[1]
            position = int(parts[-1])
            return uniprot_id, position
    except Exception as e:
        logging.error(f"Error parsing FASTA header {header}: {e}")
    return None, None

def read_fasta_file(fasta_file, label):
    """Read FASTA file and return list of entries"""
    data = []
    try:
        with open(fasta_file) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                uniprot_id, position = parse_fasta_header(record.id)
                if uniprot_id and position:
                    data.append({
                        'uniprot_id': uniprot_id,
                        'position': position,
                        'sequence': str(record.seq),
                        'label': label
                    })
        logging.info(f"Successfully read {len(data)} entries from {fasta_file}")
    except Exception as e:
        logging.error(f"Error reading FASTA file {fasta_file}: {e}")
        raise
    return data

def calculate_features(pose, position, sequence, window_size=33):
    """Calculate structural features with validation and proper padding"""
    half_window = window_size // 2
    
    # Initialize features with default values
    features = {
        'sequence': sequence,
        'phi': np.zeros(window_size),
        'psi': np.zeros(window_size),
        'omega': np.zeros(window_size),
        'tau': np.zeros(window_size),
        'chi1': np.zeros(window_size),
        'chi2': np.zeros(window_size),
        'chi3': np.zeros(window_size),
        'chi4': np.zeros(window_size),
        'sasa': np.zeros(window_size),
        'ss': ['L'] * window_size,  # Secondary structure with default 'L'
        'plDDT': np.zeros(window_size)
    }
    
    try:
        # Pre-calculate SASA and SS
        sasa_calc = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()
        sasa_calc.calculate(pose)
        residue_sasa_list = sasa_calc.get_residue_sasa()
        
        dssp = pyrosetta.rosetta.protocols.moves.DsspMover()
        dssp.apply(pose)
        
        # Fill features
        for i in range(window_size):
            pos = position - half_window + i
            if 1 <= pos <= pose.total_residue():
                residue = pose.residue(pos)

                # Backbone angles
                features['phi'][i] = pose.phi(pos)
                features['psi'][i] = pose.psi(pos)
                features['omega'][i] = pose.omega(pos)

                # Tau angle
                if 1 < pos < pose.total_residue():
                    prev_ca = pose.residue(pos-1).xyz("CA")
                    curr_ca = residue.xyz("CA")
                    next_ca = pose.residue(pos+1).xyz("CA")
                    features['tau'][i] = pyrosetta.rosetta.numeric.angle_degrees(
                        prev_ca, curr_ca, next_ca
                    )

                # Chi angles
                n_chi = residue.nchi()
                for j in range(1, 5):
                    if j <= n_chi:
                        features[f'chi{j}'][i] = residue.chi(j)

                # SASA
                features['sasa'][i] = residue_sasa_list[pos]

                # Secondary structure
                features['ss'][i] = pose.secstruct(pos)

                # plDDT
                features['plDDT'][i] = pose.pdb_info().bfactor(pos, 1)

    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        raise
        
    return features

def process_data(positive_fasta, negative_fasta, output_file="processed_features_fixed_test.csv"):
    """Process FASTA files with progress tracking and validation"""
    try:
        # Read FASTA files
        logging.info("Reading FASTA files...")
        positive_data = read_fasta_file(positive_fasta, 1)
        negative_data = read_fasta_file(negative_fasta, 0)
        all_data = positive_data + negative_data

        # Define CSV headers
        columns = ['label', 'entry', 'pos', 'sequence', 
                   'phi', 'psi', 'omega', 'tau', 
                   'chi1', 'chi2', 'chi3', 'chi4',
                   'sasa', 'ss', 'plDDT']
        
        # Create CSV file with headers
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)

        # Process each entry. Remember to update file paths as needed.
        for entry in tqdm(all_data, desc="Processing structures"):
            try:
                pdb_file = f"data/structure/test/AF-{entry['uniprot_id']}.pdb"
                if not os.path.exists(pdb_file):
                    logging.warning(f"File not found: {pdb_file}")
                    continue
                
                pose = pyrosetta.pose_from_file(pdb_file)
                features = calculate_features(pose, entry['position'], entry['sequence'])

                # Format numeric features with double quotes
                row_data = [
                    entry['label'], 
                    entry['uniprot_id'], 
                    entry['position'], 
                    features["sequence"],  # Keep sequence without quotes
                    f'{",".join(map(str, features["phi"]))}',
                    f'{",".join(map(str, features["psi"]))}',
                    f'{",".join(map(str, features["omega"]))}',
                    f'{",".join(map(str, features["tau"]))}',
                    f'{",".join(map(str, features["chi1"]))}',
                    f'{",".join(map(str, features["chi2"]))}',
                    f'{",".join(map(str, features["chi3"]))}',
                    f'{",".join(map(str, features["chi4"]))}',
                    f'{",".join(map(str, features["sasa"]))}',
                    "".join(features["ss"]),  # Keep ss as plain text without quotes
                    f'{",".join(map(str, features["plDDT"]))}'
                ]

                # Append row to CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(row_data)

            except Exception as e:
                logging.error(f"Error processing {entry['uniprot_id']} position {entry['position']}: {e}")
                continue

        return pd.read_csv(output_file, quotechar='"')

    except Exception as e:
        logging.error(f"Error in process_data: {e}")
        raise

if __name__ == "__main__":
    setup_pyrosetta()
    
    # This extracts features from the test set. Change the path to the training set if needed.
    df = process_data(
        positive_fasta="data/test/fasta/test_positive_sites.fasta",
        negative_fasta="data/test/fasta/test_negative_sites.fasta"
    )

    print("\nProcessing complete. Saved output to processed_features_fixed.csv.")
    print(f"Total entries processed: {len(df)}")