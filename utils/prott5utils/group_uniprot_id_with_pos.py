import pandas as pd

def process_both_fasta(positions_fasta, full_seq_fasta):
    # First process the positions file as before
    id_positions = {}
    
    with open(positions_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parts = line.strip().split('|')
                uniprot_id = parts[1]
                position = int(parts[-1].split('-')[-1])
                
                if uniprot_id not in id_positions:
                    id_positions[uniprot_id] = []
                id_positions[uniprot_id].append(position)
    
    # Process the full sequence file
    full_sequences = {}
    
    with open(full_seq_fasta, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id:
                    full_sequences[current_id] = ''.join(current_seq)
                
                # Get new ID
                parts = line.strip().split('|')
                current_id = parts[1]
                current_seq = []
            else:
                current_seq.append(line.strip())
        
        # Don't forget to save the last sequence
        if current_id:
            full_sequences[current_id] = ''.join(current_seq)
    
    # Create DataFrame with both information
    df = pd.DataFrame({
        'uniprot_id': list(id_positions.keys()),
        'positions': [','.join(map(str, sorted(pos))) for pos in id_positions.values()],
        'full_sequence': [full_sequences.get(uniprot_id, '') for uniprot_id in id_positions.keys()]
    })
    
    return df

# Usage
df = process_both_fasta('LMSuccSite/data/test/fasta/test_all.fasta', 'LMSuccSite/data/full_sequences/test_sequence.fasta')
df.to_csv('LMSuccSite/data/grouped/test.csv', index=False)
# print(df)