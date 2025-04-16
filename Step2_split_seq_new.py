import pandas as pd
import random
from tqdm import tqdm
import csv

# Define split length
split_length = 32  # or 64

# Load the data file
data_file_path = 'sProtein_delta_ACGT.csv'
data_df = pd.read_csv(data_file_path)
print(data_df.head())

# Output file path
output_file_path = 'sProtein_delta_ACGT_32.csv'

def split_sequence(sequence, length):
    if length <= 0 or length > len(sequence):
        raise ValueError("Invalid length. Length must be greater than 0 and less than or equal to the length of the sequence.")
    return [sequence[i:i + length] for i in range(0, len(sequence) - length + 1)]

# Open CSV for writing and write the header first
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['GeneName', 'Sub_Index', 'Family', 'Sub_Seq'])
    writer.writeheader()

    # Iterate over each row in the original dataframe with tqdm progress bar
    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing rows"):
        gene_name = row['GeneName']
        family = row['Family']
        sequence = row['Sequence']

        try:
            sub_sequences = split_sequence(sequence, split_length)
        except ValueError as e:
            print(f"Skipping GeneName: {gene_name} due to error: {e}")
            continue

        for sub_index, sub_seq in enumerate(sub_sequences):
            writer.writerow({
                'GeneName': gene_name,
                'Sub_Index': f"#{sub_index}",
                'Family': family,
                'Sub_Seq': sub_seq
            })

print("New dataset with split sequences has been written to:", output_file_path)

