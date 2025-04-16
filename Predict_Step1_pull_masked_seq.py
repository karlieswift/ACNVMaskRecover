import pandas as pd
import re

# Load the input CSV file
#input_file = "sProtein_alpha.csv"
input_file = "sProtein_delta.csv"
df = pd.read_csv(input_file)

# Initialize lists to store results
gene_names, families, sub_indices, sub_seqs, sub_seqs_n = [], [], [], [], []

# Define nucleotide pattern for search
pattern1 = re.compile(r"([ACGT]{15})([RYSWKMBDHV])([ACGT]{16})")
pattern2 = re.compile(r"([ACGT]{16})([RYSWKMBDHV])([ACGT]{15})")

# Iterate through each row in the dataframe
for idx, row in df.iterrows():
    gene_name = row['GeneName']
    family = row['Family']
    sequence = row['Sequence']

    # Search for both patterns in the sequence
    matches1 = list(pattern1.finditer(sequence))
    matches2 = list(pattern2.finditer(sequence))

    # Combine matches from both patterns
    for match_num, (match1, match2) in enumerate(zip(matches1, matches2), start=1):
        sub_seq1 = match1.group(0)
        sub_seq_n1 = re.sub(r'[RYSWKMBDHV]', 'N', sub_seq1)

        sub_seq2 = match2.group(0)
        sub_seq_n2 = re.sub(r'[RYSWKMBDHV]', 'N', sub_seq2)

        # Append the results for both patterns with the same index
        gene_names.append(gene_name)
        families.append(family)
        sub_indices.append(match_num)
        sub_seqs.append(sub_seq1)
        sub_seqs_n.append(sub_seq_n1)

        gene_names.append(gene_name)
        families.append(family)
        sub_indices.append(match_num)
        sub_seqs.append(sub_seq2)
        sub_seqs_n.append(sub_seq_n2)

# Create a new DataFrame with the results
result_df = pd.DataFrame({
    'GeneName': gene_names,
    'Family': families,
    'Sub_Index': sub_indices,
    'Sub_Seq': sub_seqs,
    'Sub_Seq_N': sub_seqs_n
})

# Save the results to a new CSV file
#output_file = "sProtein_alpha_mask.csv"
output_file = "sProtein_delta_mask.csv"
result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


