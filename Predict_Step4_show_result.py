import pandas as pd
import re

# Load the data file
f1 = "sProtein_alpha_mask_NV88_Predict.csv"
#f1 = 'sProtein_alpha_mask_NV344_Predict.csv'
#f1 = 'sProtein_delta_mask_NV88_Predict.csv'
df = pd.read_csv(f1)

# Define the special characters to look for
special_chars = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'}

# Lists to store new data structure
new_data = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    sub_seq = row['Sub_Seq']
    sub_seq_n = row['Sub_Seq_N']
    sub_seq_p = row['Sub_Seq_P']
    sub_seq_p_prob_str = row['Sub_Seq_P_Prob']
    
    # Convert the string of probabilities to a list of floats
    sub_seq_p_prob = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", sub_seq_p_prob_str)]
    
    # Find the position of the first special character in Sub_Seq
    special_char_position = -1
    for i, char in enumerate(sub_seq):
        if char in special_chars:
            special_char_position = i
            break
    
    # If no special character is found, skip this row
    if special_char_position == -1:
        continue
    
    # Split the sequences into three parts
    part1 = special_char_position
    part2 = special_char_position + 1
    part3 = special_char_position + 2
    
    sub_seq_split = sub_seq[:part1] + '-' + sub_seq[part1:part2] + '-' + sub_seq[part2:]
    sub_seq_n_split = sub_seq_n[:part1] + '-' + sub_seq_n[part1:part2] + '-' + sub_seq_n[part2:]
    sub_seq_p_split = sub_seq_p[:part1] + '-' + sub_seq_p[part1:part2] + '-' + sub_seq_p[part2:]
    
    # Get the special character probability
    special_char_prob = sub_seq_p_prob[special_char_position]
    
    # Create the Special_Char_Predict column
    #special_char_predict = f"{sub_seq[part1:part2]}->{sub_seq_n[part1:part2]}->{sub_seq_p[part1:part2]}"
    special_char_predict = f"{sub_seq[part1:part2]}--->{sub_seq_p[part1:part2]}"

    # Append the new row to the list
    new_data.append([
        row['GeneName'],
        row['Family'],
        row['Sub_Index'],
        sub_seq_split,
        sub_seq_n_split,
        sub_seq_p_split,
        special_char_predict,
        special_char_prob
    ])

# Create a new DataFrame with the new data structure
new_df = pd.DataFrame(new_data, columns=[
    'GeneName', 'Family', 'Sub_Index', 'Sub_Seq', 'Sub_Seq_N', 'Sub_Seq_P', 'Special_Char_Predict', 'Special_Char_Prob'
])

# Save the new DataFrame to a CSV file
new_df.to_csv("sProtein_alpha_mask_NV88_Predict_Results.csv", index=False)
#new_df.to_csv("sProtein_alpha_mask_NV344_Predict_Results.csv", index=False)
#new_df.to_csv("sProtein_delta_mask_NV88_Predict_Results.csv", index=False)

print("Data transformation complete and saved")

