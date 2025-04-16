
import itertools, time
import numpy as np
from collections import Counter
from common import compute_covariance, compute_covariance_py, compute_covariance_klm
from common import compute_covariance_abs, compute_covariance_klm_abs, compute_covariance_abs_py
from common import compute_covariance_sq, compute_covariance_klm_sq, compute_covariance_square_py
from common import compute_covariance_klmp, compute_covariance_klmpq
from basicSetting import NUCLEOTIDE_DICT_ORDER_LIST, ORDER_MATRIX

kernel_func_py = {0:compute_covariance_py,   # original
                  1:compute_covariance_abs_py,   # absolute
                  2:compute_covariance_square_py}  # square

# 计算非对称的2-mer的
def calculate_NV_16(sequence, mu, n):

    time_start = time.perf_counter()

    S2 = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    n_kl = {pair: S2.count(pair) for pair in set(S2)}

    time_tag_enum = time.perf_counter()

    w_kl = {}
    for pair in set(S2):
        indicator = ['0'] * len(sequence)
        for i in range(len(sequence) - 1):
            if sequence[i:i + 2] == pair:
                indicator[i] = '1'
                if i + 1 < len(sequence):
                    indicator[i + 1] = '1'
        w_kl[pair] = ''.join(indicator)

    time_01string = time.perf_counter()

    nucleotides = NUCLEOTIDE_DICT_ORDER_LIST
    covariances = {}
    for k in nucleotides:
        for l in nucleotides:
            pair = k + l
            if pair in n_kl:
                covariances[pair] = compute_covariance(k, l, sequence, mu, w_kl, n)
                # covariances[pair] = compute_covariance_sq(k, l, sequence, mu, w_kl, n)

    # Initialize all possible pairs of two characters from the list (including order)
    time_calc_cov = time.perf_counter()

    all_pairs = ORDER_MATRIX[2]
    all_pairs_val = [(covariances[t] if t in covariances else 0) for t in all_pairs]
    time_calc_cov2 = time.perf_counter()

    # Update the all_pair_cov dictionary with the given covariances

    time_update = time.perf_counter()
    print("[covariances original]", covariances)

    all_time = time_update - time_start
    print("Total time: %.4fs" % (time_update - time_start))
    print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start), "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
    print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum), "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
    print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string), "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
    print(">>Time of [covariances 2]: %.4fs" % (time_calc_cov2 - time_calc_cov), "Ratio: %.4f" % ((time_calc_cov2 - time_calc_cov) / all_time * 100), "%")
    print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov2), "Ratio: %.4f" % ((time_update - time_calc_cov2) / all_time * 100), "%")

    # Output only the values of the updated dictionary
    # return list(all_pair_cov.values())
    return all_pairs_val


# 计算非对称的2-mer的
def calculate_NV_16_py(sequence, miu, n, k=2, kernel_flag=0, print_flag=False):

    time_start = time.perf_counter()

    S = {sequence[i:i+k] for i in range(len(sequence) - 1)}

    time_tag_enum = time.perf_counter()

    w_kl = {}                                                    # initial 2-mer substring 0/1 coding
    sequence_np = np.array(list(sequence) + ["\t"])              # 尾部 padding 一个字符，用于填充 | len==70w * 1000次，32s
    single_01 = {nuc: (sequence_np==nuc) for nuc in NUCLEOTIDE_DICT_ORDER_LIST}

    for pair in S:                                               # get every 01 string of 2-mer sequence tag
        tmp_s = single_01[pair[0]][:-1] & single_01[pair[1]][1:]  # 01串与右移一个地址的自己取交，末尾必为多余的k-1个0
        tmp_s[1:] |= tmp_s[:-1]                                   # 保留第一位的必0项，得到右平移一位的01串
        w_kl[pair] = tmp_s

    time_01string = time.perf_counter()

    covariances = {tag:kernel_func_py[kernel_flag](tag, miu, w_kl, n) for tag in S}

    # for tag in S2:
    #     covariances[tag] = compute_covariance_py2(tag, miu, w_kl, n)
    time_calc_cov = time.perf_counter()

    # Initialize all possible pairs of two characters from the list (including order)
    # all_pair_cov = [0] * len(ORDER_MATRIX[2])
    # all_pair_cov = np.zeros(shape=(len(ORDER_MATRIX[2]),))

    # Update the all_pair_cov dictionary with the given covariances
    # for pair in covariances:
    #     if pair in INDEX_MATRIX[2]:
    #         all_pair_cov[INDEX_MATRIX[2][pair]] = covariances[pair]
    all_pair_cov = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[k]]
    time_update = time.perf_counter()

    if print_flag:
        print("[covariances speed up]", covariances)
        all_time = time_update - time_start
        print("Total time: %.4fs" % (time_update - time_start))
        print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start), "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
        print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum), "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
        print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string), "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
        print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov), "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")
    # Output only the values of the updated dictionary
    return all_pair_cov


def calculate_NV_16_opt(sequence, mu, n):
    # Create pairs and count their occurrences
    S2 = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    n_kl = Counter(S2)

    # Create indicators for pairs
    w_kl = {}
    for pair in n_kl:
        indicator = ['0'] * len(sequence)
        for i in range(len(sequence) - 1):
            if sequence[i:i + 2] == pair:
                indicator[i] = '1'
                if i + 1 < len(sequence):
                    indicator[i + 1] = '1'
        w_kl[pair] = ''.join(indicator)

    nucleotides = ['A', 'C', 'G', 'T', 'N']
    covariances = {}

    for k, l in itertools.product(nucleotides, repeat=2):
        pair = k + l
        if pair in n_kl:
            covariances[pair] = compute_covariance(k, l, sequence, mu, w_kl, n)

    # Initialize all possible pairs of two characters from the list (including order)
    all_pairs = [''.join(pair) for pair in itertools.product(nucleotides, repeat=2)]
    all_pair_cov = {pair: 0 for pair in all_pairs}

    # Update the all_pair_cov dictionary with the given covariances
    for pair in covariances:
        all_pair_cov[pair] = covariances[pair]

    # Output only the values of the updated dictionary
    return list(all_pair_cov.values())


# 计算非对称的3-mer的
def calculate_NV_64(sequence, mu, n):

    time_start = time.perf_counter()

    S3 = [sequence[i:i + 3] for i in range(len(sequence) - 2)]
    n_klm = {pair: S3.count(pair) for pair in set(S3)}
    time_tag_enum = time.perf_counter()

    w_klm = {}
    for pair in set(S3):
        indicator = ['0'] * len(sequence)
        for i in range(len(sequence) - 2):
            if sequence[i:i + 3] == pair:
                indicator[i] = '1'
                if i + 1 < len(sequence):
                    indicator[i + 1] = '1'
                if i + 2 < len(sequence):
                    indicator[i + 2] = '1'
        w_klm[pair] = ''.join(indicator)
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    nucleotides = ['A', 'C', 'G', 'T', 'N']
    covariances = {}
    for k in nucleotides:
        for l in nucleotides:
            for m in nucleotides:
                pair = k + l + m
                if pair in n_klm:
                    covariances[pair] = compute_covariance_klm(k, l, m, sequence, mu, w_klm, n)
                    # covariances[pair] = compute_covariance_klm_sq(k, l, m, sequence, mu, w_klm, n)

    time_calc_cov = time.perf_counter()

    up_date = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[3]]
    time_update = time.perf_counter()
    # a_64 = [cov for pair, cov in covariances.items()]
    print("[covariances original]", covariances)
    all_time = time_update - time_start
    print("Total time: %.4fs" % (time_update - time_start))
    print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start),
          "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
    print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum),
          "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
    print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string),
          "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
    print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov),
          "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")

    return up_date


def calculate_NV_64_py(sequence, miu, n, k=3, kernel_flag=0, print_flag=False):

    time_start = time.perf_counter()

    Sk = {sequence[i:i+k] for i in range(len(sequence) - k+1)}

    time_tag_enum = time.perf_counter()

    w_klm = {}                                                   # initial 3-mer substring 0/1 coding
    sequence_np = np.array(list(sequence) + ["\t"]*(k-1))       # 尾部 padding 两个字符，用于填充 | len==70w * 1000次，32s
    single_01 = {nuc: (sequence_np==nuc) for nuc in NUCLEOTIDE_DICT_ORDER_LIST}

    for pair in Sk:                                               # get every 01 string of 2-mer sequence tag
        tmp_s = single_01[pair[0]][0:-2] & single_01[pair[1]][1:-1] & single_01[pair[2]][2:]
                                                                  # 01串与右移一个地址的自己取交，末尾必为多余的k-1个0
        tmp_s[1:-1] |= tmp_s[0:-2]                                # 保留第一位的必0项，得到右平移一位的01串
        tmp_s[2:]   |= tmp_s[1:-1]
        w_klm[pair] = tmp_s
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    covariances = {tag:kernel_func_py[kernel_flag](tag, miu, w_klm, n) for tag in Sk}

    # for tag in S2:
    #     covariances[tag] = compute_covariance_py2(tag, miu, w_kl, n)
    time_calc_cov = time.perf_counter()

    # Initialize all possible pairs of two characters from the list (including order)
    # all_pair_cov = [0] * len(ORDER_MATRIX[2])
    # all_pair_cov = np.zeros(shape=(len(ORDER_MATRIX[2]),))

    # Update the all_pair_cov dictionary with the given covariances
    # for pair in covariances:
    #     if pair in INDEX_MATRIX[2]:
    #         all_pair_cov[INDEX_MATRIX[2][pair]] = covariances[pair]
    all_pair_cov = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[k]]
    time_update = time.perf_counter()

    if print_flag:
        print("[covariances speed up]", covariances)
        all_time = time_update - time_start
        print("Total time: %.4fs" % (time_update - time_start))
        print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start), "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
        print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum), "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
        print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string), "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
        print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov), "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")
    # Output only the values of the updated dictionary
    return all_pair_cov


# 计算非对称的4-mer的
def calculate_NV_256(sequence, mu, n):

    time_start = time.perf_counter()

    S4 = [sequence[i:i + 4] for i in range(len(sequence) - 3)]
    n_klmp = {pair: S4.count(pair) for pair in set(S4)}
    time_tag_enum = time.perf_counter()

    w_klmp = {}
    for pair in set(S4):
        indicator = ['0'] * len(sequence)
        for i in range(len(sequence) - 3):
            if sequence[i:i + 4] == pair:
                indicator[i] = '1'
                if i + 1 < len(sequence):
                    indicator[i + 1] = '1'
                if i + 2 < len(sequence):
                    indicator[i + 2] = '1'
                if i + 3 < len(sequence):
                    indicator[i + 3] = '1'
        w_klmp[pair] = ''.join(indicator)
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    nucleotides = ['A', 'C', 'G', 'T', 'N']
    covariances = {}
    for k in nucleotides:
        for l in nucleotides:
            for m in nucleotides:
                for p in nucleotides:
                    pair = k + l + m + p
                    if pair in n_klmp:
                        covariances[pair] = compute_covariance_klmp(k, l, m, p, sequence, mu, w_klmp, n)

    time_calc_cov = time.perf_counter()

    up_date = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[4]]
    time_update = time.perf_counter()
    # a_64 = [cov for pair, cov in covariances.items()]
    print("[covariances original]", covariances)
    all_time = time_update - time_start
    print("Total time: %.4fs" % (time_update - time_start))
    print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start),
          "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
    print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum),
          "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
    print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string),
          "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
    print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov),
          "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")

    return up_date


def calculate_NV_256_py(sequence, miu, n, k=4, kernel_flag=0, print_flag=False):

    time_start = time.perf_counter()

    Sk = {sequence[i:i+k] for i in range(len(sequence) - k + 1)}

    time_tag_enum = time.perf_counter()

    w_klm = {}                                                   # initial 3-mer substring 0/1 coding
    sequence_np = np.array(list(sequence) + ["\t"] * (k-1))      # 尾部 padding k-1个字符，用于填充 | len==70w * 1000次，32s
    single_01 = {nuc: (sequence_np==nuc) for nuc in NUCLEOTIDE_DICT_ORDER_LIST}
    # print(sequence)
    # print(sequence_np)
    # print(single_01)
    for pair in Sk:                                               # get every 01 string of 2-mer sequence tag
        tmp_s = single_01[pair[0]][0:-3] & single_01[pair[1]][1:-2] & single_01[pair[2]][2:-1] & single_01[pair[3]][3:]
                                                                  # 01串与右移一个地址的自己取交，末尾必为多余的k-1个0
        tmp_s[1:-2] |= tmp_s[:-3]                                 # 保留第一位的必0项，得到右平移一位的01串
        tmp_s[2:]   |= tmp_s[:-2]
        w_klm[pair] = tmp_s
    # print(w_klm)
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    covariances = {tag:kernel_func_py[kernel_flag](tag, miu, w_klm, n) for tag in Sk}

    # for tag in S2:
    #     covariances[tag] = compute_covariance_py2(tag, miu, w_kl, n)
    time_calc_cov = time.perf_counter()

    # Initialize all possible pairs of two characters from the list (including order)
    # all_pair_cov = [0] * len(ORDER_MATRIX[2])
    # all_pair_cov = np.zeros(shape=(len(ORDER_MATRIX[2]),))

    # Update the all_pair_cov dictionary with the given covariances
    # for pair in covariances:
    #     if pair in INDEX_MATRIX[2]:
    #         all_pair_cov[INDEX_MATRIX[2][pair]] = covariances[pair]
    all_pair_cov = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[k]]
    time_update = time.perf_counter()

    if print_flag:
        print("[covariances speed up]", covariances)
        all_time = time_update - time_start
        print("Total time: %.4fs" % (time_update - time_start))
        print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start), "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
        print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum), "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
        print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string), "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
        print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov), "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")
    # Output only the values of the updated dictionary
    return all_pair_cov

# 计算非对称的5-mer的
def calculate_NV_1024(sequence, mu, n):

    time_start = time.perf_counter()

    S5 = [sequence[i:i + 5] for i in range(len(sequence) - 4)]
    n_klmpq = {pair: S5.count(pair) for pair in set(S5)}
    time_tag_enum = time.perf_counter()

    w_klmpq = {}
    for pair in set(S5):
        indicator = ['0'] * len(sequence)
        for i in range(len(sequence) - 4):
            if sequence[i:i + 5] == pair:
                indicator[i] = '1'
                if i + 1 < len(sequence):
                    indicator[i + 1] = '1'
                if i + 2 < len(sequence):
                    indicator[i + 2] = '1'
                if i + 3 < len(sequence):
                    indicator[i + 3] = '1'
                if i + 4 < len(sequence):
                    indicator[i + 4] = '1'
        w_klmpq[pair] = ''.join(indicator)
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    nucleotides = ['A', 'C', 'G', 'T', 'N']
    covariances = {}
    for k in nucleotides:
        for l in nucleotides:
            for m in nucleotides:
                for p in nucleotides:
                    for q in nucleotides:
                        pair = k + l + m + p + q
                        if pair in n_klmpq:
                            covariances[pair] = compute_covariance_klmpq(k, l, m, p, q, sequence, mu, w_klmpq, n)

    time_calc_cov = time.perf_counter()

    up_date = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[5]]
    time_update = time.perf_counter()
    # a_64 = [cov for pair, cov in covariances.items()]
    print("[covariances original]", covariances)
    all_time = time_update - time_start
    print("Total time: %.4fs" % (time_update - time_start))
    print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start),
          "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
    print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum),
          "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
    print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string),
          "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
    print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov),
          "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")

    return up_date


def calculate_NV_1024_py(sequence, miu, n, k=5, kernel_flag=0, print_flag=False):

    time_start = time.perf_counter()

    Sk = {sequence[i:i+k] for i in range(len(sequence) - k + 1)}

    time_tag_enum = time.perf_counter()

    w_klm = {}                                                   # initial 3-mer substring 0/1 coding
    sequence_np = np.array(list(sequence) + ["\t"] * (k-1))      # 尾部 padding k-1个字符，用于填充 | len==70w * 1000次，32s
    single_01 = {nuc: (sequence_np==nuc) for nuc in NUCLEOTIDE_DICT_ORDER_LIST}

    for pair in Sk:                                               # get every 01 string of 2-mer sequence tag
        tmp_s = single_01[pair[0]][0:-4] & single_01[pair[1]][1:-3] & single_01[pair[2]][2:-2] & \
                single_01[pair[3]][3:-1] & single_01[pair[4]][4:]
                                                                  # 01串与右移一个地址的自己取交，末尾必为多余的k-1个0
        tmp_s[1:-3] |= tmp_s[0:-4]                                # 保留第一位的必0项，得到右平移一位的01串
        tmp_s[2:-2] |= tmp_s[1:-3]
        tmp_s[3:]   |= tmp_s[1:-2]

        w_klm[pair] = tmp_s
    # ----------------------------------------------------------------------------
    time_01string = time.perf_counter()

    covariances = {tag:kernel_func_py[kernel_flag](tag, miu, w_klm, n) for tag in Sk}
    # print(covariances)
    # for tag in S2:
    #     covariances[tag] = compute_covariance_py2(tag, miu, w_kl, n)
    time_calc_cov = time.perf_counter()

    # Initialize all possible pairs of two characters from the list (including order)
    # all_pair_cov = [0] * len(ORDER_MATRIX[2])
    # all_pair_cov = np.zeros(shape=(len(ORDER_MATRIX[2]),))

    # Update the all_pair_cov dictionary with the given covariances
    # for pair in covariances:
    #     if pair in INDEX_MATRIX[2]:
    #         all_pair_cov[INDEX_MATRIX[2][pair]] = covariances[pair]
    all_pair_cov = [covariances[tag] if tag in covariances else 0 for tag in ORDER_MATRIX[k]]
    time_update = time.perf_counter()

    if print_flag:
        print("[covariances speed up]", covariances)
        all_time = time_update - time_start
        print("Total time: %.4fs" % (time_update - time_start))
        print(">>Time of [tag enumerate]: %.4fs" % (time_tag_enum - time_start), "Ratio: %.4f" % ((time_tag_enum - time_start) / all_time * 100), "%")
        print(">>Time of [calc 01string]: %.4fs" % (time_01string - time_tag_enum), "Ratio: %.4f" % ((time_01string - time_tag_enum) / all_time * 100), "%")
        print(">>Time of [covariances 1]: %.4fs" % (time_calc_cov - time_01string), "Ratio: %.4f" % ((time_calc_cov - time_01string) / all_time * 100), "%")
        print(">>Time of [update tagCov]: %.4fs" % (time_update - time_calc_cov), "Ratio: %.4f" % ((time_update - time_calc_cov) / all_time * 100), "%")

    return all_pair_cov

