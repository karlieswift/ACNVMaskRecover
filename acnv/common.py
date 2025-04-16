
from collections import defaultdict
import numpy as np
from basicSetting import MAX_DNA_LEN, NUCLEOTIDE_DICT_ORDER_LIST

# initial maximum i*i table

print("[Initial] initial number table...")

TABLE_N_1 = [i for i in range(MAX_DNA_LEN)]
print("[Initial] initial process:  1/12...")

TABLE_N_2 = [i*i for i in range(MAX_DNA_LEN)]
print("[Initial] initial process:  2/12...")

TABLE_N_3 = [i*t for i, t in enumerate(TABLE_N_2)]
print("[Initial] initial process:  3/12...")

TABLE_N_4 = [i*t for i, t in enumerate(TABLE_N_3)]
print("[Initial] initial process:  4/12...")

TABLE_N_5 = [i*t for i, t in enumerate(TABLE_N_4)]
print("[Initial] initial process:  5/12...")

TABLE_N_6 = [i*t for i, t in enumerate(TABLE_N_5)]
print("[Initial] initial process:  6/12...")
#
# TABLE_N_7 = [i*t for i, t in enumerate(TABLE_N_6)]
# print("[Initial] initial process:  7/20...")
#
# TABLE_N_8 = [i*t for i, t in enumerate(TABLE_N_7)]
# print("[Initial] initial process:  8/20...")
#
# TABLE_N_9 = [i*t for i, t in enumerate(TABLE_N_8)]
# print("[Initial] initial process:  9/20...")
#
# TABLE_N_X = [i*t for i, t in enumerate(TABLE_N_9)]
# print("[Initial] initial process: 10/20...")

TABLE_N_1 = np.array(TABLE_N_1, dtype=object)
print("[Initial] initial process:  7/12...")

TABLE_N_2 = np.array(TABLE_N_2, dtype=object)
print("[Initial] initial process:  8/12...")

TABLE_N_3 = np.array(TABLE_N_3, dtype=object)
print("[Initial] initial process:  9/12...")

TABLE_N_4 = np.array(TABLE_N_4, dtype=object)
print("[Initial] initial process: 10/12...")

TABLE_N_5 = np.array(TABLE_N_5, dtype=object)
print("[Initial] initial process: 11/12...")

TABLE_N_6 = np.array(TABLE_N_6, dtype=object)
print("[Initial] initial process: 12/12...")
#
# TABLE_N_7 = np.array(TABLE_N_7, dtype=object)
# print("[Initial] initial process: 17/20...")
#
# TABLE_N_8 = np.array(TABLE_N_8, dtype=object)
# print("[Initial] initial process: 18/20...")
#
# TABLE_N_9 = np.array(TABLE_N_9, dtype=object)
# print("[Initial] initial process: 19/20...")
#
# TABLE_N_X = np.array(TABLE_N_X, dtype=object)
# print("[Initial] initial process: 20/20...")
print("[Initial] initial finished...")

# for high-dimentional moment calc
TABLE_N_1_TO_6 = [None, TABLE_N_1, TABLE_N_2, TABLE_N_3, TABLE_N_4, TABLE_N_5, TABLE_N_6]

# common calculation
def calculate_values(sequence):
    nucleotides = 'ACGTN'
    n = {nucl: 0 for nucl in nucleotides}
    mu = {nucl: 0 for nucl in nucleotides}
    D2 = {nucl: 0 for nucl in nucleotides}

    for nucl in nucleotides:
        n[nucl] = sequence.count(nucl)

    for nucl in nucleotides:
        positions = [i + 1 for i, char in enumerate(sequence) if char == nucl]
        if positions:
            mu[nucl] = sum(positions) / n[nucl]

    total_length = len(sequence)
    for nucl in nucleotides:
        positions = [i + 1 for i, char in enumerate(sequence) if char == nucl]
        if positions:
            D2[nucl] = sum((pos - mu[nucl]) ** 2 for pos in positions) / total_length / n[nucl]

    return n, mu, D2


# a no usage func?
def k_mer_natural_vector(sequence, k=2):
    N = len(sequence) - k + 1
    k_mers = defaultdict(list)
    for i in range(N):
        k_mer = sequence[i:i + k]
        k_mers[k_mer].append(i + 1)

    counts = {}
    means = {}
    second_moments = {}

    for k_mer, positions in k_mers.items():
        counts[k_mer] = len(positions)
        if positions:
            mean_position = np.mean(positions)
            means[k_mer] = mean_position
            second_moments[k_mer] = np.mean([(p - mean_position) ** 2 / N for p in positions])
        else:
            means[k_mer] = 0
            second_moments[k_mer] = 0

    natural_vector = []
    for k_mer in sorted(k_mers.keys()):
        natural_vector.extend([counts[k_mer], means[k_mer], second_moments[k_mer]])

    nv = list(counts.values()) + list(means.values()) + list(second_moments.values())
    return nv


# kernel func of old-version asymmetric natural vector
def compute_covariance(k, l, sequence, mu_, w_kl, n_):
    n = len(sequence)
    indices = np.arange(1, n + 1)
    cov_sum = sum((i - mu_[k]) * (i - mu_[l]) * int(w_kl.get(k + l, '0')[i - 1]) for i in indices if i <= len(w_kl.get(k + l, '0')))
    return cov_sum / (n_[k] * n_[l])


# kernel func of old-version asymmetric natural vector
def compute_covariance_abs(k, l, sequence, mu_, w_kl, n_):
    n = len(sequence)
    indices = np.arange(1, n + 1)
    cov_sum = sum(abs((i - mu_[k]) * (i - mu_[l]) * int(w_kl.get(k + l, '0')[i - 1])) for i in indices if i <= len(w_kl.get(k + l, '0')))
    return cov_sum / (n_[k] * n_[l])


# kernel func of old-version asymmetric natural vector
def compute_covariance_sq(k, l, sequence, mu_, w_kl, n_):
    n = len(sequence)
    indices = np.arange(1, n + 1)
    cov_sum = sum(((i - mu_[k]) * (i - mu_[l]) * int(w_kl.get(k + l, '0')[i - 1]))**2 for i in indices if i <= len(w_kl.get(k + l, '0')))
    return cov_sum / ((n_[k] * n_[l])**2)


# [new version] sub function to calc the cov containing sigma, when k == 2
def sub_sigma_2(tag, index_one, miu_, n_):

    k, l = tag
    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_2[index_one])             # sigma i*i
    # part_2 = np.sum(index_one) * (miu_[k] + miu_[l])  # sigma i*miu[k] + i*miu[l]
    # part_3 = len(index_one) * miu_[k] * miu_[l]       # sigma miu[k]*miu[l]
    cov_sum = np.sum(TABLE_N_2[index_one]) - np.sum(index_one) * (miu_[k] + miu_[l]) + len(index_one) * miu_[k] * miu_[l]

    return cov_sum / (n_[k] * n_[l])


def sub_sigma_abs_2(tag, index_one, miu_, n_):

    k, l = tag
    # original calc
    # cov_sum = sum(abs((i - miu_[k]) * (i - miu_[l]) * int(tmp_get[i - 1])) for i in indices if i <= len(tmp_get))

    # part_1[pos] = np.sum(TABLE_N_2[index_one])             # sigma i*i + miu[k]*miu[l]
    # part_2[neg] = np.sum(index_one) * (miu_[k] + miu_[l])  # sigma i*miu[k] + i*miu[l]
    # when part1 > part2: part1 - part2
    # when part2 > part1: part2 - part1
    part1 = TABLE_N_2[index_one] + miu_[k] * miu_[l]
    part2 = TABLE_N_1[index_one]

    neg_index = part2 > part1  # negative; ~neg_index (list) is aiming the positive item

    part1 -= part2
    cov_sum = np.sum(part1[~neg_index]) - np.sum(part1[neg_index])

    # cov_sum = np.sum(TABLE_N_2[index_one]) - np.sum(index_one) * (miu_[k] + miu_[l]) + len(index_one) * miu_[k] * miu_[l]

    return cov_sum / (n_[k] * n_[l])


def sub_sigma_sq_2(tag, index_one, miu_, n_):
    k, l = tag
    # original calc
    # cov_sum = sum(((i - miu_[k])**2 * (i - miu_[l])**2 * int(tmp_get[i - 1])) for i in indices if i <= len(tmp_get))
    # that means:
    # sum the square of original sub item(as a list)
    cov_vector = TABLE_N_2[index_one] - index_one * (miu_[k] + miu_[l]) + miu_[k] * miu_[l]
    cov_sum = np.dot(cov_vector.T, cov_vector)
    # print(cov_sum[0][0])
    return cov_sum[0][0] / (n_[k] * n_[k] * n_[l] * n_[l])

# [new version] sub function to calc the cov containing sigma, when k == 3
def sub_sigma_3(tag, index_one, miu_, n_):

    k, l, m = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_3[index_one])                                                 # sigma i**3 * ...
    # part_2 = np.sum(TABLE_N_2[index_one]) * (miu_[k] + miu_[l] + miu_[m])                 # sigma i**2 * ...
    # part_3 = np.sum(index_one) * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[l]*miu_[m])    # sigma i**1 * ...
    # part_4 = len(index_one) * miu_[k] * miu_[l] * miu_[m]                                 # sigma miu[k]*miu[l]*miu[m]

    cov_sum = np.sum(TABLE_N_3[index_one]) \
              - np.sum(TABLE_N_2[index_one]) * (miu_[k] + miu_[l] + miu_[m]) \
              + np.sum(index_one) * (miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[l] * miu_[m]) \
              - len(index_one) * (miu_[k] * miu_[l] * miu_[m])

    return cov_sum / (n_[k] * n_[l] * n_[m])


def sub_sigma_abs_3(tag, index_one, miu_, n_):

    k, l, m = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_3[index_one])                                                 # sigma i**3 * ...
    # part_2 = np.sum(TABLE_N_2[index_one]) * (miu_[k] + miu_[l] + miu_[m])                 # sigma i**2 * ...
    # part_3 = np.sum(index_one) * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[l]*miu_[m])    # sigma i**1 * ...
    # part_4 = len(index_one) * miu_[k] * miu_[l] * miu_[m]                                 # sigma miu[k]*miu[l]*miu[m]

    part1 = TABLE_N_3[index_one] + TABLE_N_1[index_one]* (miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[l] * miu_[m])
    part2 = TABLE_N_2[index_one] * (miu_[k] + miu_[l] + miu_[m]) + miu_[k] * miu_[l] * miu_[m]

    neg_index = part2 > part1
    part1 -= part2

    cov_sum = np.sum(part1[~neg_index]) - np.sum(part1[neg_index])

    return cov_sum / (n_[k] * n_[l] * n_[m])


def sub_sigma_sq_3(tag, index_one, miu_, n_):
    k, l, m = tag
    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_3[index_one])                                                 # sigma i**3 * ...
    # part_2 = np.sum(TABLE_N_2[index_one]) * (miu_[k] + miu_[l] + miu_[m])                 # sigma i**2 * ...
    # part_3 = np.sum(index_one) * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[l]*miu_[m])    # sigma i**1 * ...
    # part_4 = len(index_one) * miu_[k] * miu_[l] * miu_[m]                                 # sigma miu[k]*miu[l]*miu[m]

    # that means:
    # sum the square of original sub item(as a list)
    # cov_vector = TABLE_N_2[index_one] - index_one * (miu_[k] + miu_[l]) + miu_[k] * miu_[l]
    cov_vector = TABLE_N_3[index_one] - TABLE_N_2[index_one] * (miu_[k] + miu_[l] + miu_[m]) \
              + index_one * (miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[l] * miu_[m]) - miu_[k] * miu_[l] * miu_[m]
    cov_sum = np.dot(cov_vector.T, cov_vector)

    return cov_sum[0][0] / ((n_[k] * n_[l] * n_[m])**2)


# [new version] sub function to calc the cov containing sigma, when k == 4
def sub_sigma_4(tag, index_one, miu_, n_):
    k, l, m, p = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))
    # part_1 = np.sum(TABLE_N_4[index_one])                                                 # sigma i**4 * ...
    # part_2 = np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**3 * ...
    # part_3 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(4,2)...)                     # sigma i**2 * ...
    # part_4 = np.sum(index_one) * (CombineMultiply(4,3)...)                                # sigma i**1 * ...
    # part_5 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu[k]*[l]*[m]*[p]

    v = miu_[k] * miu_[l] * miu_[m] * miu_[p]    # multiply value of miu_k, l , m, p
    cov_sum = np.sum(TABLE_N_4[index_one]) \
              - np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) \
              + np.sum(TABLE_N_2[index_one]) * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[k]*miu_[p] + miu_[l]*miu_[m] + miu_[l]*miu_[p] + miu_[m]*miu_[p])\
              - np.sum(index_one) * (v/miu_[k] + v/miu_[l] + v/miu_[m] + v/miu_[p]) \
              + len(index_one) * v

    # print("####" * 9)
    # print("1/n\t\t", (n_[k] * n_[l] * n_[m] * n_[p]))
    # print(cov_sum, cov_sum / (n_[k] * n_[l] * n_[m] * n_[p]))
    # print("cov_sum1\t", cov_sum / (n_[k] * n_[l] * n_[m] * n_[p]))
    # cov_sum2 = np.sum(TABLE_N_4[index_one]) / (n_[k] * n_[l] * n_[m] * n_[p]) \
    #           - np.sum(TABLE_N_3[index_one]) / (n_[k] * n_[l] * n_[m] * n_[p]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) \
    #           + np.sum(TABLE_N_2[index_one]) / (n_[k] * n_[l] * n_[m] * n_[p]) * (
    #                       miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[k] * miu_[p] + miu_[l] * miu_[m] + miu_[l] *
    #                       miu_[p] + miu_[m] * miu_[p]) \
    #           - np.sum(index_one) / (n_[k] * n_[l] * n_[m] * n_[p]) * (v / miu_[k] + v / miu_[l] + v / miu_[m] + v / miu_[p]) \
    #           + len(index_one) / (n_[k] * n_[l] * n_[m] * n_[p]) * v
    # print("cov_sum2\t", cov_sum2)
    # cov_sum3 = np.sum(TABLE_N_4[index_one]) / (n_[k] * n_[l] * n_[m] * n_[p]) \
    #          - np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) / (n_[k] * n_[l] * n_[m] * n_[p]) \
    #          + np.sum(TABLE_N_2[index_one]) * (
    #                      miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[k] * miu_[p] + miu_[l] * miu_[m] + miu_[l] * miu_[
    #                  p] + miu_[m] * miu_[p]) / (n_[k] * n_[l] * n_[m] * n_[p]) \
    #          - np.sum(index_one) * (v / miu_[k] + v / miu_[l] + v / miu_[m] + v / miu_[p]) / (n_[k] * n_[l] * n_[m] * n_[p]) \
    #          + len(index_one) * v / (n_[k] * n_[l] * n_[m] * n_[p])
    # print("cov_sum3\t", cov_sum3)
    # print("####" * 9)
    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p])


def sub_sigma_abs_4(tag, index_one, miu_, n_):
    k, l, m, p = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_4[index_one])                                                 # sigma i**4 * ...
    # part_2 = np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**3 * ...
    # part_3 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(4,2)...)                     # sigma i**2 * ...
    # part_4 = np.sum(index_one) * (CombineMultiply(4,3)...)                                # sigma i**1 * ...
    # part_5 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu[k]*[l]*[m]*[p]

    # part1: part_1 + part_3 + part_5
    # part2: part_2 + part_4
    v = miu_[k] * miu_[l] * miu_[m] * miu_[p]    # multiply value of miu_k, l , m, p
    part1 = TABLE_N_4[index_one] + \
            TABLE_N_2[index_one] * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[k]*miu_[p] + miu_[l]*miu_[m] + miu_[l]*miu_[p] + miu_[m]*miu_[p])\
            + v

    part2 = TABLE_N_3[index_one] * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) + \
            TABLE_N_1[index_one] * (v/miu_[k] + v/miu_[l] + v/miu_[m] + v/miu_[p])

    neg_index = part2 > part1
    part1 -= part2
    cov_sum = np.sum(part1[~neg_index]) - np.sum(part1[neg_index])

    # cov_sum1 = np.sum(TABLE_N_4[index_one]) \
    #           - np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) \
    #           + np.sum(TABLE_N_2[index_one]) * (miu_[k]*miu_[l] + miu_[k]*miu_[m] + miu_[k]*miu_[p] + miu_[l]*miu_[m] + miu_[l]*miu_[p] + miu_[m]*miu_[p])\
    #           - np.sum(index_one) * (v/miu_[k] + v/miu_[l] + v/miu_[m] + v/miu_[p]) \
    #           + len(index_one) * v
    # print("abs:", cov_sum, cov_sum / (n_[k] * n_[l] * n_[m] * n_[p]))
    # print("ori:", cov_sum1, cov_sum1 / (n_[k] * n_[l] * n_[m] * n_[p]))
    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p])


def sub_sigma_sq_4(tag, index_one, miu_, n_):
    k, l, m, p = tag
    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))
    # part_1 = np.sum(TABLE_N_4[index_one])                                                 # sigma i**4 * ...
    # part_2 = np.sum(TABLE_N_3[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**3 * ...
    # part_3 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(4,2)...)                     # sigma i**2 * ...
    # part_4 = np.sum(index_one) * (CombineMultiply(4,3)...)                                # sigma i**1 * ...
    # part_5 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu[k]*[l]*[m]*[p]
    # that means:
    # sum the square of original sub item(as a list)
    v = miu_[k] * miu_[l] * miu_[m] * miu_[p]  # multiply value of miu_k, l , m, p
    cov_vector = TABLE_N_4[index_one] - TABLE_N_3[index_one] * (miu_[k] + miu_[l] + miu_[m] + miu_[p]) \
              + TABLE_N_2[index_one] * (miu_[k] * miu_[l] + miu_[k] * miu_[m] + miu_[k] * miu_[p] + miu_[l] * miu_[m] + miu_[l] * miu_[p] + miu_[m] * miu_[p]) \
              - index_one * (v / miu_[k] + v / miu_[l] + v / miu_[m] + v / miu_[p]) + v

    cov_sum = np.dot(cov_vector.T, cov_vector)

    return cov_sum[0][0] / ((n_[k] * n_[l] * n_[m] * n_[p])**2)


# [new version] sub function to calc the cov containing sigma, when k == 5
def sub_sigma_5(tag, index_one, miu_, n_):
    k, l, m, p, q = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_5[index_one])                                                 # sigma i**5 * ...
    # part_2 = np.sum(TABLE_N_4[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**4 * ...
    # part_3 = np.sum(TABLE_N_3[index_one]) * (CombineMultiply(5,2)...)                     # sigma i**3 * ...
    # part_4 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(5,3)...)                     # sigma i**2 * ...
    # part_5 = np.sum(index_one) * (CombineMultiply(5,4)...)                                # sigma i**1 * ...
    # part_6 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu k*l*m*p*q

    v5 = (miu_[k] * miu_[l] * miu_[m] * miu_[p] * miu_[q])  # multiply value of miu_k, l , m, p, q
    v2_list = [miu_[k] * miu_[l], miu_[k] * miu_[m], miu_[k] * miu_[p], miu_[k] * miu_[q], miu_[p] * miu_[q],
               miu_[l] * miu_[m], miu_[l] * miu_[p], miu_[l] * miu_[q], miu_[m] * miu_[p], miu_[m] * miu_[q]]
    # print(v5)
    # print("====" * 9)
    # print(v2_list)
    # print(sum(v2_list))
    # print("===="*9)
    # np.dtype
    cov_sum = np.sum(TABLE_N_5[index_one]) \
              - np.sum(TABLE_N_4[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p] + miu_[q]) \
              + np.sum(TABLE_N_3[index_one]) * sum(v2_list) \
              - np.sum(TABLE_N_2[index_one]) * sum(v5 / t for t in v2_list) \
              + np.sum(index_one) * (v5 / miu_[k] + v5 / miu_[l] + v5 / miu_[m] + v5 / miu_[p] + v5 / miu_[q]) \
              - len(index_one) * v5
    # print("sum(list(TABLE_N_5[index_one]))", sum(list(TABLE_N_5[index_one])))
    # print("sum(list(TABLE_N_4[index_one]))", sum(list(TABLE_N_4[index_one])))
    # print("sum(list(TABLE_N_3[index_one]))", sum(list(TABLE_N_3[index_one])))
    # print("sum(list(TABLE_N_2[index_one]))", sum(list(TABLE_N_2[index_one])))
    # print(sum(index_one))
    # print(cov_sum)
    # exit()
    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p] * n_[q])


def sub_sigma_abs_5(tag, index_one, miu_, n_):
    k, l, m, p, q = tag

    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_5[index_one])                                                 # sigma i**5 * ...
    # part_2 = np.sum(TABLE_N_4[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**4 * ...
    # part_3 = np.sum(TABLE_N_3[index_one]) * (CombineMultiply(5,2)...)                     # sigma i**3 * ...
    # part_4 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(5,3)...)                     # sigma i**2 * ...
    # part_5 = np.sum(index_one) * (CombineMultiply(5,4)...)                                # sigma i**1 * ...
    # part_6 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu k*l*m*p*q

    v5 = (miu_[k] * miu_[l] * miu_[m] * miu_[p] * miu_[q])  # multiply value of miu_k, l , m, p, q
    v2_list = [miu_[k] * miu_[l], miu_[k] * miu_[m], miu_[k] * miu_[p], miu_[k] * miu_[q], miu_[p] * miu_[q],
               miu_[l] * miu_[m], miu_[l] * miu_[p], miu_[l] * miu_[q], miu_[m] * miu_[p], miu_[m] * miu_[q]]

    # part1: part_1 + part_3 + part_5
    # part2: part_2 + part_4 + part_6
    part1 = TABLE_N_5[index_one] + \
            TABLE_N_3[index_one] * sum(v2_list) + \
            TABLE_N_1[index_one] * (v5 / miu_[k] + v5 / miu_[l] + v5 / miu_[m] + v5 / miu_[p] + v5 / miu_[q])

    part2 = TABLE_N_4[index_one] * (miu_[k] + miu_[l] + miu_[m] + miu_[p] + miu_[q]) + \
            TABLE_N_2[index_one] * sum(v5 / t for t in v2_list) + v5

    neg_index = part2 > part1
    part1 -= part2
    cov_sum = np.sum(part1[~neg_index]) - np.sum(part1[neg_index])
    # np.dtype
    # cov_sum = np.sum(TABLE_N_5[index_one]) \
    #           - np.sum(TABLE_N_4[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[p] + miu_[q]) \
    #           + np.sum(TABLE_N_3[index_one]) * sum(v2_list) \
    #           - np.sum(TABLE_N_2[index_one]) * sum(v5 / t for t in v2_list) \
    #           + np.sum(index_one) * (v5 / miu_[k] + v5 / miu_[l] + v5 / miu_[m] + v5 / miu_[p] + v5 / miu_[q]) \
    #           - len(index_one) * v5
    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p] * n_[q])


def sub_sigma_sq_5(tag, index_one, miu_, n_):
    k, l, m, p, q = tag
    # original calc
    # cov_sum = sum((i - miu_[k]) * (i - miu_[l]) * (i - miu_[m]) * (i - miu_[p])\
    #  * int(tmp_get[i - 1]) for i in indices if i <= len(tmp_get))

    # part_1 = np.sum(TABLE_N_5[index_one])                                                 # sigma i**5 * ...
    # part_2 = np.sum(TABLE_N_4[index_one]) * (miu_[k] + miu_[l] + miu_[m] + miu_[m])       # sigma i**4 * ...
    # part_3 = np.sum(TABLE_N_3[index_one]) * (CombineMultiply(5,2)...)                     # sigma i**3 * ...
    # part_4 = np.sum(TABLE_N_2[index_one]) * (CombineMultiply(5,3)...)                     # sigma i**2 * ...
    # part_5 = np.sum(index_one) * (CombineMultiply(5,4)...)                                # sigma i**1 * ...
    # part_6 = len(index_one) * miu_[k] * miu_[l] * miu_[m] * miu_[p]                       # sigma miu k*l*m*p*q

    # that means:
    # sum the square of original sub item(as a list)
    v5 = (miu_[k] * miu_[l] * miu_[m] * miu_[p] * miu_[q])  # multiply value of miu_k, l , m, p, q
    v2_list = [miu_[k] * miu_[l], miu_[k] * miu_[m], miu_[k] * miu_[p], miu_[k] * miu_[q], miu_[p] * miu_[q],
               miu_[l] * miu_[m], miu_[l] * miu_[p], miu_[l] * miu_[q], miu_[m] * miu_[p], miu_[m] * miu_[q]]

    cov_vector = TABLE_N_5[index_one] - TABLE_N_4[index_one] * (miu_[k] + miu_[l] + miu_[m] + miu_[p] + miu_[q]) \
              + TABLE_N_3[index_one] * sum(v2_list) - TABLE_N_2[index_one] * sum(v5 / t for t in v2_list) \
              + index_one * (v5 / miu_[k] + v5 / miu_[l] + v5 / miu_[m] + v5 / miu_[p] + v5 / miu_[q]) - v5

    cov_sum = np.dot(cov_vector.T, cov_vector)

    return cov_sum[0][0] / ((n_[k] * n_[l] * n_[m] * n_[p] * n_[q])**2)


# [new version] function point list, easy to use it!
sub_sigma = [None, None, sub_sigma_2, sub_sigma_3, sub_sigma_4, sub_sigma_5]

sub_sigma_abs = [None, None, sub_sigma_abs_2, sub_sigma_abs_3, sub_sigma_abs_4, sub_sigma_abs_5]

sub_sigma_sq  = [None, None, sub_sigma_sq_2, sub_sigma_sq_3, sub_sigma_sq_4, sub_sigma_sq_5]

def compute_covariance_py(tag, miu_, w_dict, n_):
    ...
    tmp_get = w_dict[tag] if tag in w_dict else None
    if tmp_get is None:
        return 0.0

    # the list of index whose value equal to 1, and add 1 all(start idx is 1)
    index_one = np.argwhere((tmp_get == 1)) + 1

    # calculate asymmetric (partial) vector
    return sub_sigma[len(tag)](tag, index_one, miu_, n_)


def compute_covariance_abs_py(tag, miu_, w_dict, n_):
    ...
    tmp_get = w_dict[tag] if tag in w_dict else None
    if tmp_get is None:
        return 0.0

    # the list of index whose value equal to 1, and add 1 all(start idx is 1)
    index_one = np.argwhere((tmp_get == 1)) + 1

    # calculate asymmetric (partial) vector
    return sub_sigma_abs[len(tag)](tag, index_one, miu_, n_)


def compute_covariance_square_py(tag, miu_, w_dict, n_):
    ...
    tmp_get = w_dict[tag] if tag in w_dict else None
    if tmp_get is None:
        return 0.0

    # the list of index whose value equal to 1, and add 1 all(start idx is 1)
    index_one = np.argwhere((tmp_get == 1)) + 1

    # calculate asymmetric (partial) vector
    return sub_sigma_sq[len(tag)](tag, index_one, miu_, n_)


def compute_covariance_klm(k, l, m, sequence, mu_, w_klm, n_):
    n = len(sequence)
    # print(k, l, m, n, mu_)
    indices = np.arange(1, n + 1)
    # print(indices)
    # print(len(w_klm.get(k + l + m, '0')))
    # for i in indices:
    #    print(int(w_klm.get(k + l + m, '0')[i - 1]))
    cov_sum = sum((i - mu_[k]) * (i - mu_[l]) * (i - mu_[m]) * int(w_klm.get(k + l + m, '0')[i - 1]) for i in indices if
                  i <= len(w_klm.get(k + l + m, '0')))
    # print(cov_sum)
    return cov_sum / (n_[k] * n_[l] * n_[m])
    # return cov_sum / sqrt(n_[k]) / sqrt(n_[l]) / sqrt(n_[m]) / n


def compute_covariance_klm_abs(k, l, m, sequence, mu_, w_klm, n_):
    n = len(sequence)
    # print(k, l, m, n, mu_)
    indices = np.arange(1, n + 1)
    # print(indices)
    # print(len(w_klm.get(k + l + m, '0')))
    # for i in indices:
    #    print(int(w_klm.get(k + l + m, '0')[i - 1]))
    cov_sum = sum(abs((i - mu_[k]) * (i - mu_[l]) * (i - mu_[m]) * int(w_klm.get(k + l + m, '0')[i - 1])) for i in indices if
                  i <= len(w_klm.get(k + l + m, '0')))
    # print(cov_sum)
    return cov_sum / (n_[k] * n_[l] * n_[m])
    # return cov_sum / sqrt(n_[k]) / sqrt(n_[l]) / sqrt(n_[m]) / n


def compute_covariance_klm_sq(k, l, m, sequence, mu_, w_klm, n_):
    n = len(sequence)
    # print(k, l, m, n, mu_)
    indices = np.arange(1, n + 1)
    # print(indices)
    # print(len(w_klm.get(k + l + m, '0')))
    # for i in indices:
    #    print(int(w_klm.get(k + l + m, '0')[i - 1]))
    cov_sum = sum(((i - mu_[k]) * (i - mu_[l]) * (i - mu_[m]) * int(w_klm.get(k + l + m, '0')[i - 1]))**2 for i in indices if
                  i <= len(w_klm.get(k + l + m, '0')))
    # print(cov_sum)
    return cov_sum / ((n_[k] * n_[l] * n_[m])**2)
    # return cov_sum / sqrt(n_[k]) / sqrt(n_[l]) / sqrt(n_[m]) / n


def compute_covariance_klmp(k, l, m, p, sequence, mu_, w_klmp, n_):
    n = len(sequence)
    indices = np.arange(1, n + 1)

    cov_sum = sum((i - mu_[k]) * (i - mu_[l]) * (i - mu_[m]) * (i - mu_[p]) * int(w_klmp.get(k + l + m + p, '0')[i - 1]) for i in indices if
                  i <= len(w_klmp.get(k + l + m + p, '0')))

    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p])


def compute_covariance_klmpq(k, l, m, p, q, sequence, mu_, w_klmpq, n_):
    n = len(sequence)
    indices = np.arange(1, n + 1)

    cov_sum = sum((i - mu_[k]) * (i - mu_[l]) * (i - mu_[m]) * (i - mu_[p]) * (i - mu_[q]) * \
                  int(w_klmpq.get(k + l + m + p + q, '0')[i - 1]) for i in indices if
                  i <= len(w_klmpq.get(k + l + m + p + q, '0')))

    return cov_sum / (n_[k] * n_[l] * n_[m] * n_[p] * n_[q])


#################################################
# 定义计算各阶矩的函数
def calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k):
    moments = {k: [0, 0, 0, 0, 0] for k in range(2, max_k + 1)}
    for i, nt in enumerate(sequence):
        for j, k in enumerate(NUCLEOTIDE_DICT_ORDER_LIST):
            if nt == k:
                for n in range(2, max_k + 1):
                    # moments[n][j] += (((i + 1) - avg_positions[j]) ** n) / ((seq_len ** (n - 1)) * (nucleotide_counts[j] ** (n - 1)))
                    moments[n][j] += (((i + 1) - avg_positions[j]) ** n) / (nucleotide_counts[j] ** n)
    return moments

def calculate_k(sequence, max_k):
    # Function to calculate nucleotide counts, average positions, and moments
    nu1, nu2, nu3, nu4 = NUCLEOTIDE_DICT_ORDER_LIST
    nucleotide_counts = [sequence.count(nu1), sequence.count(nu2), sequence.count(nu3), sequence.count(nu4)]
    seq_len = len(sequence)
    avg_positions = []
    for nt in NUCLEOTIDE_DICT_ORDER_LIST:
        positions = [(i + 1) for i, base in enumerate(sequence) if base == nt]
        avg_positions.append(sum(positions) / len(positions) if positions else 0)

    # Calculate higher-order moments
    moments = calculate_moments(sequence, avg_positions, seq_len, nucleotide_counts, max_k)
    nucleotide_vector = nucleotide_counts + avg_positions
    for k in range(2, max_k + 1):
        nucleotide_vector += moments[k]

    return nucleotide_vector

# 加速版本：定义计算各阶矩的函数
# moments: [n][j]   n: n-square, j: j-th nucleotide(A, C, G, T)
# sum of (  (index+1 - avrg) / count  ) ** n
# [attention] 20240810注释掉，未调试完毕，仅参考，勿用
# def calculate_k_mod(sequence, max_k):
#     # Function to calculate nucleotide counts, average positions, and moments
#     sequence_np = np.array(list(sequence))
#     single_01 = {nuc: (sequence_np == nuc) for nuc in NUCLEOTIDE_DICT_ORDER_LIST}
#
#     nu1, nu2, nu3, nu4 = NUCLEOTIDE_DICT_ORDER_LIST
#     nucleotide_counts = [np.sum(single_01[nu1]), np.sum(single_01[nu2]), np.sum(single_01[nu3]), np.sum(single_01[nu4])]
#
#     avg_positions = []
#     moments = {k: [0, 0, 0, 0, 0] for k in range(2, max_k + 1)}
#     for j, nt in enumerate(NUCLEOTIDE_DICT_ORDER_LIST):
#         index_one = np.argwhere((single_01[nt] == 1)) + 1
#         if len(index_one) > 0:
#             avg_positions.append(np.sum(index_one) / len(index_one) if len(index_one) > 0 else 0)
#             # Calculate higher-order moments
#             tmp = index_one - avg_positions[-1]
#             t = tmp * 1
#             for n in range(2, max_k + 1):
#                 t = t * tmp
#                 moments[n][j] = np.sum(t)
#         else:
#             avg_positions.append(0)
#             for n in range(2, max_k+1):
#                 moments[n][j] = 0
#
#     nucleotide_vector = nucleotide_counts + avg_positions
#     for k in range(2, max_k + 1):
#         nucleotide_vector += moments[k]
#
#     return nucleotide_vector



def write_list_to_file(file_path, data_list):
    try:
        with open(file_path, 'w') as file:
            formatted_data = ' '.join(f"{item:.2f}" for item in data_list)
            file.write(formatted_data)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

