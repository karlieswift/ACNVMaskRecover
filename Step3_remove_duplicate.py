import pandas as pd

# Load the dataset
#file_path = 'sProtein_alpha_ACGT_64.csv'
file_path = 'sProtein_delta_ACGT_32.csv'
file_df = pd.read_csv(file_path)

# Display the first few rows and shape before filtering
print("Before filtering:")
print(file_df.head())
print(file_df.shape)

# Filter out duplicate sequences, keeping only the first occurrence
filtered_df = file_df.drop_duplicates(subset='Sub_Seq', keep='first')

# Display the first few rows and shape after filtering
print("After filtering:")
print(filtered_df.head())
print(filtered_df.shape)

# Optionally, save the filtered dataset to a new CSV file
#filtered_df.to_csv('sProtein_alpha_ACGT_64_nodup.csv', index=False)
filtered_df.to_csv('sProtein_delta_ACGT_32_nodup.csv', index=False)
