import requests
import sys
import os
from tqdm import tqdm

def extract_id_from_header(header):
    """Extract identifier from FASTA header"""
    header = header[1:].strip()
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1].strip()
    return header.strip()

def download_alphafold_pdb(uniprot_id, output_dir):
    """Download AlphaFold PDB file for a given UniProt ID"""
    # Create URL for the PDB file
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"AF-{uniprot_id}.pdb")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading {uniprot_id}: {e}")
        return False

def count_entries(filename):
    """Count number of entries in FASTA file"""
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

# Adjust the output directory as needed
def process_fasta_file(filename, output_dir="data/structure/train"):
    """Process FASTA file and download AlphaFold PDBs"""
    successful_downloads = []
    failed_downloads = []
    total_entries = count_entries(filename)
    
    print(f"\nDownloading AlphaFold PDB files to: {output_dir}")
    print("Processing FASTA entries...")
    
    with tqdm(total=total_entries, desc="Downloading PDBs", unit="entry") as pbar:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    uniprot_id = extract_id_from_header(line)
                    pbar.set_description(f"Downloading {uniprot_id}")
                    
                    if download_alphafold_pdb(uniprot_id, output_dir):
                        successful_downloads.append(uniprot_id)
                    else:
                        failed_downloads.append(uniprot_id)
                    
                    pbar.update(1)
    
    return successful_downloads, failed_downloads

def main(fasta_file):
    print(f"Processing file: {fasta_file}")
    successful, failed = process_fasta_file(fasta_file)
    
    print("\nDownload Summary:")
    print("-" * 50)
    print(f"Successfully downloaded: {len(successful)} PDB files")
    print(f"Failed downloads: {len(failed)} entries")
    
    if failed:
        print("\nFailed downloads for:")
        for id in failed:
            print(id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <fasta_file>")
        sys.exit(1)
    
    main(sys.argv[1])