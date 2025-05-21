def clean_csv(input_path, output_path):
    print(f"Cleaning CSV file: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Read and clean the file
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        # Skip the header line
        next(infile)
        
        # Process each line
        for line in infile:
            # Remove [ and ] and write to output
            cleaned_line = line.replace('[', '').replace(']', '')
            outfile.write(cleaned_line)
    
    print("Done! File has been cleaned.")

# File paths
input_csv = "/home/ubuntu/data/hai/thesis/get_embedded/embeddings_prott5_window_test.csv"
output_csv = "/home/ubuntu/data/hai/thesis/get_embedded/embeddings_prott5_window_test_cleaned.csv"

# Run the cleaning
clean_csv(input_csv, output_csv)