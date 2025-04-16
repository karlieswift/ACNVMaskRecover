
# the most basic setting is in there
...
# 最大基因序列长度上限设置，billion  十亿(太慢了)   10 million  一千万
# MAX_DNA_LEN = int(1e7)
MAX_DNA_LEN = int(14782125)
"""
基因有多长
这是个难以给出确定答案的问题，
基因由ATGC四种碱基组成，每个碱基成为1个bp，
有的基因很长，目前最长的基因是DMD基因，全长2,220,291bp（来自NCBI）
最短的基因比较难说，一般认为在几十bp到几百bp，
人体2万多个基因的平均长度在10-15kb左右。
"""

NUCLEOTIDE_DICT_ORDER_LIST = ["A", "C", "G", "T", "N"]  # user defined order

VALID_CHARS = {'A', 'C', 'G', 'T', 'N', 'a', 'c', 'g', 't', 'n'}

# default: dictionary order
# 人为规定好的顺序是唯一的，不会因为特殊的系统、python版本等意外情况而改变

MIN_K_OF_MER = 2
# default: 2

MAX_K_OF_MER = 5
# default: 5

# vector order setting (with different k)
ORDER_MATRIX = [None for _ in range(MAX_K_OF_MER+1)]  # ORDER_MATRIX[k] get the list of k-mer tag in user defined order
# eg. 2 -> 5
# 0 None
# 1 None
# 2 [...]  list of 2 nucleotides combination(4 ** 2 =   16)
# 3 [...]  list of 3 nucleotides combination(4 ** 3 =   64)
# 4 [...]  list of 4 nucleotides combination(4 ** 4 =  256)
# 5 [...]  list of 5 nucleotides combination(4 ** 5 = 1024)

# inverted index of k-mer partial sequence
INDEX_MATRIX = [None for _ in range(MAX_K_OF_MER+1)]  # INDEX_MATRIX[k][k-mer tag] get the index of the k-mer tag T_k
# eg. 2 -> 5
# 0 None
# 1 None
# 2 {2-mer:index...}  list of 2 nucleotides combination(4 ** 2 =   16)
# 3 {3-mer:index...}  list of 3 nucleotides combination(4 ** 3 =   64)
# 4 {4-mer:index...}  list of 4 nucleotides combination(4 ** 4 =  256)
# 5 {5-mer:index...}  list of 5 nucleotides combination(4 ** 5 = 1024)
# eg.
# AA / AAA / AAAA / AAAAA --> 0
# AC / AAC / AAAC / AAAAC --> 1
# AG / AAG / AAAG / AAAAG --> 2
# AT / AAT / AAAT / AAAAT --> 3
# ...

print("\n[CONSTRUCTION] START")
print("[CONSTRUCTION] construct k-mer matrix info:  0 / 2 ...")
tmp_order = [t for t in NUCLEOTIDE_DICT_ORDER_LIST]
for length in range(2, MAX_K_OF_MER+1):
    ...
    new_tmp_order = []
    for item in tmp_order:
        new_tmp_order += [item + t for t in NUCLEOTIDE_DICT_ORDER_LIST]

    # not smaller than setting minist k value
    if length >= MIN_K_OF_MER:
        ORDER_MATRIX[length] = new_tmp_order

    tmp_order = new_tmp_order

del tmp_order
print("[CONSTRUCTION] construct k-mer matrix info:  1 / 2 ...")
for length in range(2, MAX_K_OF_MER+1):

    temp_order_list = ORDER_MATRIX[length]

    if not temp_order_list:
        continue

    temp_dict = dict()
    for i, t in enumerate(temp_order_list):
        temp_dict[t] = i
    INDEX_MATRIX[length] = temp_dict

print("[CONSTRUCTION] construct k-mer matrix info:  2 / 2 ...")
print("[CONSTRUCTION] DONE\n\n")
