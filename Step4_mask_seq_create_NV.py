#step3-paralle version: from SplitSequences create NV s_NV_24 for example
import os
import sys
import csv
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.optimize import linprog
#from math import sqrt
import itertools
from time import time
from collections import Counter
from multiprocessing import Pool, cpu_count
import pandas as pd
import random

from acnv.common import calculate_values, calculate_k
from acnv.basicSetting import VALID_CHARS
#from originalNV import calculate_covariance
from acnv.asymmetricNV import calculate_NV_16, calculate_NV_16_py, calculate_NV_64, calculate_NV_64_py
from acnv.asymmetricNV import calculate_NV_256_py, calculate_NV_1024_py

#################################################
# 定义计算各阶矩的函数
def calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k):
    moments = {k: [0, 0, 0, 0, 0] for k in range(2, max_k + 1)}
    for i, nt in enumerate(sequence):
        for j, k in enumerate("ACGTN"):
            if nt == k:
                for n in range(2, max_k + 1):
                    #moments[n][j] += (((i + 1) - avg_positions[j]) ** n) / ((seq_len ** (n - 1)) * (nucleotide_counts[j] ** (n - 1)))
                    moments[n][j] += (((i + 1) - avg_positions[j]) ** n) / (nucleotide_counts[j] ** n)
    return moments

def calculate_k(sequence, max_k):
    # Function to calculate nucleotide counts, average positions, and moments
    nucleotide_counts = [sequence.count("A"), sequence.count("C"), sequence.count("G"), sequence.count("T"), sequence.count("N")]
    seq_len = len(sequence)
    avg_positions = []
    for nt in "ACGTN":
        positions = [(i + 1) for i, base in enumerate(sequence) if base == nt]
        avg_positions.append(sum(positions) / len(positions) if positions else 0)

    # Calculate higher-order moments
    moments = calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k)
    nucleotide_vector = nucleotide_counts + avg_positions
    for k in range(2, max_k + 1):
        nucleotide_vector += moments[k]

    return nucleotide_vector
#############################################
# 计算移除的indexs
def get_index(kmer):
    res = []
    elements = ['A', 'G', 'C', 'T', 'N']
    for i in range(kmer):
        combinations = list(itertools.product(elements, repeat=i + 1))
        for combination in combinations:
            res.append(''.join(combination))
    res = res[:5] + ['A', 'G', 'C', 'T', 'N'] + res[5:]
    indexs = []

    for i in range(len(res)):
        if 'N' in res[i]:
            indexs.append(i + 1)

    # print(len(res)-len(indexs))
    return indexs

def calculate_NV(sequence, nv_dim):
    n, mu, D2 = calculate_values(sequence)
    NV_8 = list(n.values()) + list(mu.values())  # NV_8 = [nA, nG, nC, nT, muA, muG, muC, muT]
    #NV_12 = NV_8 + list(D2.values())             # D2_4 = [D2A, D2G, D2C, D2T]
   
    
    NV_16   = calculate_NV_16_py(sequence, mu, n, kernel_flag=1) #0, default 1: abs; 2:square
    NV_64   = calculate_NV_64_py(sequence, mu, n, kernel_flag=1)
    #NV_256  = calculate_NV_256_py(sequence, mu, n, kernel_flag=1)
    #NV_1024 = calculate_NV_1024_py(sequence, mu, n, kernel_flag=1)
 
    NV_24   = NV_8 + NV_16
    NV_88   = NV_24 + NV_64
    #NV_344  = NV_88 + NV_256
    #NV_1368 = NV_344 + NV_1024
    
    #NV_32 = calculate_k(sequence, 7) #define how many j


    if nv_dim == 'NV_344':
        remove_index= get_index(kmer=4)
        NV_344_modified = [value for index, value in enumerate(NV_344) if (index + 1) not in remove_index]
        return NV_344_modified
    if nv_dim == 'NV_1368':
        remove_index= get_index(kmer=5)
        NV_1368_modified = [value for index, value in enumerate(NV_1368) if (index + 1) not in remove_index]
        return NV_1368_modified

    if nv_dim == 'NV_88':
        #print(NV_88)
        #print(len(NV_88))
        # Specify the indices to remove (1-based)
        remove_index = (5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35, 40, 45, 50, 55, 56, 57, 58, 59, 60, 65, 70, 75, 80, 81, 82, 83, 84, 85, 90, 95, 100, 105, 106, 107, 108, 109, 110, 115, 120, 125, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160)
        #print(remove_index)
        remove_index= get_index(kmer=3)
        #print(remove_index)
        # Adjust to 0-based index by subtracting 1
        NV_88_modified = [value for index, value in enumerate(NV_88) if (index + 1) not in remove_index]
        #print(len(NV_88_modified))
        return NV_88_modified
    elif nv_dim == 'NV_24':
        print(NV_24)
        print(len(NV_24))
        # Specify the indices to remove (1-based)
        remove_index = (5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35)
        # Adjust to 0-based index by subtracting 1
        NV_24_modified = [value for index, value in enumerate(NV_24) if (index + 1) not in remove_index]
        print(len(NV_24_modified))
        return NV_24_modified
    elif nv_dim == 'NV_32':
        print(NV_32)
        print(len(NV_32))
        # Remove every 5th element
        NV_32_modified = [value for index, value in enumerate(NV_32) if (index + 1) % 5 != 0]
        print(NV_32_modified)
        return NV_32_modified
    else:
        return [0, 0, 0, 0, 0]

def mask_positions(sub_seq, num_mask):
    seq_length = len(sub_seq)
    
    # Define the range for valid positions (excluding the first 15 and last 15 positions)
    # so if the len is 32, only the middle 2 position can be masked
    valid_positions = range(15, seq_length - 15)
    #valid_positions = range(31, seq_length - 31) #if len is 64, 31 NN 31
    # Generate all combinations of positions to mask from the valid positions
    position_combinations = itertools.combinations(valid_positions, num_mask)
    
    # Generate all combinations of positions to mask
    #position_combinations = itertools.combinations(range(seq_length), num_mask)

    '''
    sample_size = 2 #200  # Desired number of random samples
    # Generate all possible combinations of masked positions
    all_combinations = list(itertools.combinations(range(seq_length), num_mask))
    # Randomly sample the combinations without replacement
    position_combinations = random.sample(all_combinations, min(sample_size, len(all_combinations)))
    '''

    masked_sequences = []
    for positions in position_combinations:
        masked_seq = list(sub_seq)  # Convert sequence to list for mutable changes
        for pos in positions:
            masked_seq[pos] = 'N'  # Replace the selected positions with 'N'
        masked_sequences.append("".join(masked_seq))  # Convert back to string and add to the list

    return masked_sequences

def process_row(row):
    gene_name = row['GeneName']
    sub_index = row['Sub_Index']
    family = row['Family']
    sub_seq = row['Sub_Seq']

    results = []  # List to store all the results

    # Calculate NV_Masked for the original Sub_Seq
    NV_Masked_original = calculate_NV(sub_seq, 'NV_88')
    results.append([gene_name, sub_index + '_H', family, sub_seq, sub_seq] + NV_Masked_original)

    for m in range(1, 2): # only mask 1 # m will be 1 or 2
        masked_seqs_ = mask_positions(sub_seq, m)  # Generate masked sequences with 'm' masks
        # Remove duplicates by converting to a set
        masked_seqs = list(set(masked_seqs_))

        i = 0
        for masked_seq in masked_seqs[:]:  # Iterate over each masked sequence
            # Call the calculate_NV function with the masked sequence
            NV_Masked = calculate_NV(masked_seq, 'NV_88')

            # Create the submask_index (convert m and i to strings for concatenation)
            submask_index = f"{sub_index}_{m}_{i}"

            # Append the record to the results list
            results.append([gene_name, submask_index, family, masked_seq, sub_seq] + NV_Masked)
            i += 1  # Increment i

    return results  # Return the combined list of all records

def write_results_to_csv(results, writer):
    for result in results:
        writer.writerow(result)

#file_path = 'sProtein_alpha_ACGT_32_nodup.csv'
file_path = 'sProtein_delta_ACGT_32_nodup.csv'
file_df = pd.read_csv(file_path)
print(file_df.head())
print(file_df.shape)

NV_dim = 88
# Open output CSV file and write headers
s_NV_file = open('sProtein_delta_ACGT_32_nodup_NV88.csv', 'w', newline='')
s_NV_writer = csv.writer(s_NV_file)
s_NV_writer.writerow(['GeneName'] + ['Sub_Index'] + ['Family'] + ['Mask_Seq']  + ['Sub_Seq'] + [f'V{i+1}' for i in range(NV_dim)])

# Define the number of processes to use
num_processes = cpu_count()

# Use Pool to parallelize the computation
with Pool(num_processes) as pool:
    for index, results in enumerate(pool.imap(process_row, file_df.to_dict('records'))):
        if index % 100000 == 0:
            print("Processing: ", index)
        write_results_to_csv(results, s_NV_writer)

s_NV_file.close()

