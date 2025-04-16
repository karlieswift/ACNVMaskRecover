import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.saving import register_keras_serializable
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个 GPU
'''
# 获取所有的物理 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置每个 GPU 的内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 内存增长必须在初始化 GPU 之前设置
        print(e)
'''
# Nucleotide to integer and back mapping
nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
num_classes = 4
NV_Len = 88#1368
epoch = 50
lr = 0.00005
batch = 128#32
seq_length = 32#64

#from tensorflow.keras.mixed_precision import set_global_policy
#set_global_policy('mixed_float16')

# Load the dataset
#file_path = 'sProtein_delta_ACGT_32_nodup_NV88.csv'
file_path = 'sProtein_alpha_ACGT_32_nodup_NV88.csv'
df = pd.read_csv(file_path, index_col=0)
print(df.shape)

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the length of Sub_Seq for each row and filter out rows with length not equal to seq_length
#df = df[df['Sub_Seq'].apply(len) == seq_length]
#print(df.shape)

# Extract features and target
features = df.iloc[:, -NV_Len:].values  # Last 88 columns as source data

# Convert Sub_Seq to list of integers using mapping
sequences = df['Sub_Seq'].apply(lambda x: [nucleotide_to_int[nuc] for nuc in x])
# Print the first few sequences to check the conversion
#print(sequences.head())
    
# One-hot encode the sequences
one_hot_sequences = np.array([to_categorical(seq, num_classes=num_classes) for seq in sequences])

# Release memory
del sequences

# Ensure all sequences are of the same length
sequence_length = one_hot_sequences.shape[1]

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(features, one_hot_sequences, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Convert NumPy arrays to DataFrames
X_train_df = pd.DataFrame(X_train)
X_val_df = pd.DataFrame(X_val)
X_test_df = pd.DataFrame(X_test)

# Save splits to new files
#X_train_df.to_csv('train_data.csv', index=False)
#X_val_df.to_csv('eval_data.csv', index=False)
#X_test_df.to_csv('test_data.csv', index=False)

# Reshape features for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Release memory
del features, one_hot_sequences
# Release memory
del X_temp, y_temp

# Build LSTM model #old 256
model = Sequential()
model.add(LSTM(1024, input_shape=(NV_Len, 1)))
model.add(RepeatVector(seq_length))
model.add(LSTM(1024, return_sequences=True))
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

# Use a lower learning rate and gradient clipping
optimizer = Adam(learning_rate=lr, clipnorm=1.0)

@register_keras_serializable()
# Custom loss function
def custom_loss(y_true, y_pred):
    # Standard categorical crossentropy loss
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    return cross_entropy_loss
    pass

# Compile the model using the custom loss function
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='alpha_best_model_88_32_mask.h5',
    #filepath='delta_best_model_88_32_mask.h5',
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only when the validation loss improves
    save_weights_only=False,  # Save the entire model, not just the weights
    mode='min',  # Save when the monitored metric is minimized
    verbose=1  # Print a message when the model is saved
)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# Train the model with the checkpoint callback
history = model.fit(
    X_train, y_train,
    epochs=epoch,
    batch_size=batch,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint_callback]
)

# Release memory
del X_train, y_train, X_val, y_val

from tensorflow.keras import backend as K
K.clear_session()

# Load the best model before evaluation
best_model = load_model('alpha_best_model_88_32_mask.h5')
#best_model = load_model('delta_best_model_88_32_mask.h5')

# Evaluate the model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions
predictions = best_model.predict(X_test)

# Convert predictions and y_test from one-hot encoding to nucleotide sequences
predicted_nucleotides = np.argmax(predictions, axis=-1)
original_nucleotides = np.argmax(y_test, axis=-1)

# Convert integers back to nucleotide characters
predicted_sequences = [''.join([int_to_nucleotide[nuc] for nuc in seq]) for seq in predicted_nucleotides]
original_sequences = [''.join([int_to_nucleotide[nuc] for nuc in seq]) for seq in original_nucleotides]

# Calculate the error rate
total_nucleotides = len(predicted_sequences) * seq_length
incorrect_nucleotides = sum(p != o for p_seq, o_seq in zip(predicted_sequences, original_sequences) for p, o in zip(p_seq, o_seq))
error_rate = incorrect_nucleotides / total_nucleotides

print(f'Error Rate: {error_rate:.4f}')

# Print original and predicted nucleotide sequences for the entire test dataset
print(f"Total test dataset: {len(X_test)}")
#for i in range(len(X_test)):
#    print(f"Original sequence {i + 1}: {original_sequences[i]}")
#    print(f"Predicted sequence {i + 1}: {predicted_sequences[i]}")

# Compare each character for each record in X_test and report
cc = 0
for idx, (orig_seq, pred_seq) in enumerate(zip(original_sequences, predicted_sequences)):
    cc += 1
    match_count = 0
    mismatch_count = 0
    for orig_char, pred_char in zip(orig_seq, pred_seq):
        if orig_char == pred_char:
            match_count += 1
        else:
            mismatch_count += 1

    total_characters = match_count + mismatch_count
    accuracy = match_count / total_characters
    mismatch_rate = mismatch_count / total_characters

    print(f'Record {idx + 1}:')
    print(f"Original  sequence {idx + 1}: {orig_seq}")
    print(f"Predicted sequence {idx + 1}: {pred_seq}")
    print(f'Total characters compared: {total_characters}')
    print(f'Matches: {match_count}')
    print(f'Mismatches: {mismatch_count}')
    print(f'Character-level Accuracy: {accuracy:.4f}')
    print(f'Character-level Mismatch Rate: {mismatch_rate:.4f}')
    if cc == 10:
        break

# Function to calculate average metrics and maximum mismatches
def calculate_metrics(original_sequences, predicted_sequences):
    total_matches = 0
    total_mismatches = 0
    total_characters = 0
    max_mismatches = 0
    max_mismatch_record = None

    for orig_seq, pred_seq in zip(original_sequences, predicted_sequences):
        match_count = sum(1 for orig_char, pred_char in zip(orig_seq, pred_seq) if orig_char == pred_char)
        mismatch_count = len(orig_seq) - match_count

        total_matches += match_count
        total_mismatches += mismatch_count
        total_characters += len(orig_seq)

        if mismatch_count > max_mismatches:
            max_mismatches = mismatch_count
            max_mismatch_record = (orig_seq, pred_seq)

    average_matches = total_matches / len(original_sequences)
    average_mismatches = total_mismatches / len(original_sequences)
    accuracy = total_matches / total_characters
    mismatch_rate = total_mismatches / total_characters

    return {
        "average_matches": average_matches,
        "average_mismatches": average_mismatches,
        "accuracy": accuracy,
        "mismatch_rate": mismatch_rate,
        "max_mismatches": max_mismatches,
        "max_mismatch_record": max_mismatch_record
    }

# Calculate and print the metrics
metrics = calculate_metrics(original_sequences, predicted_sequences)

print(f'\nAverage Matches: {metrics["average_matches"]:.2f}')
print(f'Average Mismatches: {metrics["average_mismatches"]:.2f}')
print(f'Overall Accuracy: {metrics["accuracy"]:.4f}')
print(f'Overall Mismatch Rate: {metrics["mismatch_rate"]:.4f}')
print(f'Maximum Mismatches: {metrics["max_mismatches"]}')
print(f'Original  sequence with maximum mismatches: {metrics["max_mismatch_record"][0]}')
print(f'Predicted sequence with maximum mismatches: {metrics["max_mismatch_record"][1]}')

