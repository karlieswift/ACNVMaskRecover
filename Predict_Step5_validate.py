import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('sProtein_alpha_mask_NV88_Predict_Results.csv')
#df = pd.read_csv('sProtein_alpha_mask_NV344_Predict_Results.csv')
#df = pd.read_csv('sProtein_delta_mask_NV88_Predict_Results.csv')

# Define the patterns to validate
patterns = [
    re.compile(r'R--->(A|G)'),
    re.compile(r'Y--->(C|T)'),
    re.compile(r'S--->(G|C)'),
    re.compile(r'W--->(A|T)'),
    re.compile(r'K--->(G|T)'),
    re.compile(r'M--->(A|C)'),
    re.compile(r'B--->(C|G|T)'),
    re.compile(r'D--->(A|G|T)'),
    re.compile(r'H--->(A|C|T)'),
    re.compile(r'V--->(A|C|G)')
]

# Function to validate the pattern
def validate_special_char_predict(special_char):
    return any(pattern.match(special_char) for pattern in patterns)

# Apply the validation function to the Special_Char_Predict column
df_validated = df[df['Special_Char_Predict'].apply(validate_special_char_predict)]
df_invalid = df[~df['Special_Char_Predict'].apply(validate_special_char_predict)]

# Print the count of valid patterns/invalid
print(f"Count of valid patterns: {len(df_validated)}")
print(f"Count of invalid patterns: {len(df_invalid)}")
print(f"valid %: {len(df_validated)/(len(df_validated) + len(df_invalid))}")

# Print detailed invalid patterns
if not df_invalid.empty:
    print("Invalid patterns:")
    print(df_invalid)

# Optionally, save the validated rows to a new CSV file
df_validated.to_csv('sProtein_alpha_mask_NV88_Predict_validated.csv', index=False)
df_invalid.to_csv('sProtein_alpha_mask_NV88_Predict_invalid.csv', index=False)
#df_validated.to_csv('sProtein_alpha_mask_NV344_Predict_validated.csv', index=False)
#df_invalid.to_csv('sProtein_alpha_mask_NV344_Predict_invalid.csv', index=False)
#df_validated.to_csv('sProtein_delta_mask_NV88_Predict_validated.csv', index=False)
#df_invalid.to_csv('sProtein_delta_mask_NV88_Predict_invalid.csv', index=False)

