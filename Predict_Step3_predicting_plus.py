import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tqdm import tqdm

# Nucleotide to integer and back mapping
nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
NV_Len = 88

@register_keras_serializable()
# Custom loss function
def custom_loss(y_true, y_pred):
    # Standard categorical crossentropy loss
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cross_entropy_loss

# Load the best model
#model_path = 'alpha_best_model_88_32_mask.h5'
model_path = 'best_model_88_32_mask.h5'#old alpha model
#model_path = 'delta_best_model_88_32_mask.h5'
best_model = load_model(model_path)

# Load the data file
data_file = 'sProtein_alpha_mask_NV88.csv'
#data_file = 'sProtein_delta_mask_NV88.csv'
df = pd.read_csv(data_file)

# Prepare to store results
results = []

# Iterate through each row and make predictions with progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    # Extract the features (last 88 columns)
    input_values = row[-NV_Len:].values.astype(float)
    input_array = np.array(input_values).reshape((1, NV_Len, 1))
    
    # Make prediction
    prediction = best_model.predict(input_array, verbose=0)
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    predicted_sequence = ''.join([int_to_nucleotide[nuc] for nuc in predicted_indices])
    predicted_probability = np.max(prediction, axis=-1)[0]

    # Append the results
    results.append([
        row['GeneName'],
        row['Family'],
        row['Sub_Index'],
        row['Sub_Seq'],
        row['Sub_Seq_N'],
        predicted_sequence,
        predicted_probability
    ])

# Create a DataFrame from the results
columns = ['GeneName', 'Family', 'Sub_Index', 'Sub_Seq', 'Sub_Seq_N', 'Sub_Seq_P', 'Sub_Seq_P_Prob']
results_df = pd.DataFrame(results, columns=columns)

# Save the results to a CSV file
output_file = 'sProtein_alpha_mask_NV88_Predict_old.csv'
#output_file = 'sProtein_delta_mask_NV88_Predict.csv'
results_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
