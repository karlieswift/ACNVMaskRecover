import pandas as pd

def filter_acgt_chars(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
                
    # Filter out characters that are not 'A', 'C', 'G', 'T' in the Sequence column
    df['Sequence'] = df['Sequence'].apply(lambda x: ''.join([char for char in x if char in 'ACGT']))
                            
    # Save the filtered data to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
#filter_acgt_chars('sProtein_alpha.csv', 'sProtein_alpha_ACGT.csv')
filter_acgt_chars('sProtein_delta.csv', 'sProtein_delta_ACGT.csv')
