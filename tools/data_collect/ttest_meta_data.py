import numpy as np
import scipy.stats as stats


meta_on = """
& -0.0588 & 0.0777 & 0.0027 & 0.03 & 0.0245 & 0.0152 
&-0.0505 & 0.105 & 0.0316 & 0.0474 & 0.0379 & 0.0343
& 0.0566 & 0.0401 & 0.0226 & 0.0288 & 0.0146 & 0.0325
& -0.1511 & 0.1823 & 0.0168 & 0.0379  & 0.0716 & 0.0315
& 0.016 & 0.0455 & 0.0049 & 0.0141 & 0.0223 & 0.0205
& -0.0778 & 0.1931 & 0.0086 & -0.0173 & 0.114 & 0.0441  
"""

meta_off = """
& 0.0012 & 0.0091 & 0.0045 & -0.0062 & -0.0027 & 0.0012 
& 0.0124 & 0.1351 & 0.1162 & 0.1353 & 0.0682 & 0.0934
& 0.3715  & 0.3551 & -0.1155 & 0.1266 & 0.2115 & 0.1898
& -0.0069 & 0.0045 & 0.0019 & -0.0003 & 0.0064 & 0.0011 
& 0.1291 & 0.0164 & 0.0758 & 0.0402 & -0.105 & 0.0313 
& -0.0577 & 0.1475 & 0.0062 & 0.0066 & 0.0148 & 0.0235
"""


def clean_data(data):
    data = data.strip("\n")
    lines = data.split("\n")
    data = [line.replace("&", ",").strip(",") for line in lines]
    data = [[float(num) for num in line.split(",")] for line in data]
    data_arr = np.array(data)
    return data_arr


def compute_p_value(modality1, modality2):
    p_value_list = []
    for i in range(6):
        t_stat, p_value = stats.ttest_rel(modality1[:, i], modality2[:, i])
        p_value_list.append(p_value)
    print(p_value_list)


def print_latex_table(modality1, modality2):
    for i in range(6):
        t_stat, p_value = stats.ttest_rel(modality1[:, i], modality2[:, i])
        print(f"{p_value:.6f} & ", end="")


meta_on = clean_data(meta_on)
meat_off = clean_data(meta_off)

print_latex_table(meta_on, meat_off)

