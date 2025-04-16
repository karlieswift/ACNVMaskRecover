# step 2: from CombinedFamilySequence Split create SplitSequences
import pandas as pd
import random
from tqdm import tqdm

#define split_length
split_length = 32#64

# Load the data file
#data_file_path = 'sProtein_alpha_ACGT.csv'
data_file_path = 'sProtein_delta_ACGT.csv'
data_df = pd.read_csv(data_file_path)
print(data_df.head())

def split_sequence(sequence, length):
    # Check if the specified length is valid
    if length <= 0 or length > len(sequence):
        raise ValueError("Invalid length. Length must be greater than 0 and less than or equal to the length of the sequence.")

    # Create a list to store all sub-sequences
    sub_sequences = []

    # Iterate over the sequence and extract sub-sequences of the specified length
    for i in range(0, len(sequence) - length + 1):
        sub_sequence = sequence[i:i + length]
        sub_sequences.append(sub_sequence)

    #print(f"Length of all sub_sequences: {len(sub_sequences)}")
    return sub_sequences

def split_sequence_(sequence, part_length=128, random_sample_percentage=10):
    sequence_length = len(sequence)

    # Split sequence into parts of the specified length
    parts = [sequence[i:i + part_length] for i in range(0, sequence_length, part_length) if len(sequence[i:i + part_length]) == part_length]

    # Generate a set of indices for the parts to avoid overlap
    existing_indices = set(range(0, sequence_length, part_length))

    # Calculate the number of additional parts to generate based on the input percentage
    num_additional_parts = max(1, (sequence_length * random_sample_percentage) // part_length)

    additional_parts = []
    if sequence_length > part_length:
        while len(additional_parts) < num_additional_parts:
            start_idx = random.randint(0, sequence_length - part_length)
            if start_idx not in existing_indices and len(sequence[start_idx:start_idx + part_length]) == part_length:        
                additional_part = sequence[start_idx:start_idx + part_length]
                additional_parts.append(additional_part)
                existing_indices.add(start_idx)

    # Combine the original parts and the additional random parts
    all_parts = parts + additional_parts

    # Print out the details
    print(f"Length of sequence: {sequence_length}")
    print(f"Number of parts from initial split: {len(parts)}")
    print(f"Number of additional random parts: {len(additional_parts)}")

    return all_parts

# Initialize an empty list to hold the new data
new_data = []
'''
# Iterate over each row in the original dataframe
for index, row in data_df.iterrows():
    gene_name = row['GeneName']
    family = row['Family']
    sequence = row['Sequence']
    
    # Split the sequence into parts
    #sub_sequences = split_sequence(sequence, split_length, 8) # 1-4 good selection of random samples
    sub_sequences = split_sequence(sequence, split_length)

    # Append each part to the new data list with the appropriate format
    for sub_index, sub_seq in enumerate(sub_sequences):
        new_data.append({
            'GeneName': f"{gene_name}",
            'Sub_Index': f"#{sub_index}",
            'Family': family,
            'Sub_Seq': sub_seq
        })
'''

# Iterate over each row in the original dataframe with tqdm progress bar
for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing rows"):
    gene_name = row['GeneName']
    family = row['Family']
    sequence = row['Sequence']

    # Split the sequence into parts
    #sub_sequences = split_sequence(sequence, split_length)
    try:
        # Split the sequence into parts
        sub_sequences = split_sequence(sequence, split_length)
    except ValueError as e:
        print(f"Skipping GeneName: {gene_name} due to error: {e}")
        continue

    # Append each part to the new data list with the appropriate format
    for sub_index, sub_seq in enumerate(sub_sequences):
        new_data.append({
            'GeneName': gene_name,
            'Sub_Index': f"#{sub_index}",
            'Family': family,
            'Sub_Seq': sub_seq
        })

# Create a new dataframe from the new data list
new_data_df = pd.DataFrame(new_data)

# Write the new dataframe to a CSV file
#output_file_path = 'sProtein_alpha_ACGT_64.csv'
output_file_path = 'sProtein_delta_ACGT_32.csv'
new_data_df.to_csv(output_file_path, index=False)

print("New dataset with split sequences has been written to:", output_file_path)


