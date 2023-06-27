import numpy as np
import scipy.stats as stats

tmp_ip = """
& 0.3248 & 0.3601 & 0.3601 & 0.2120 & 0.3352 & 0.3185 
& 0.0256 & 0.0320 & 0.0185 & 0.0105 & 0.0184 & 0.0210   
& 0.4427 & 0.4767 & 0.4998 & 0.3230 & 0.4675 & 0.4420 
& 0.6216 & 0.6753 & 0.6836 & 0.5228 & 0.6456 & 0.6298
"""

spa_ip = """
& 0.5923 & 0.6912 & 0.6436 & 0.5195 & 0.6273 & 0.6148
& 0.5882 & 0.6550 & 0.6326 & 0.5003 & 0.6199 & 0.5992 
& 0.5693 & 0.6254 & 0.6070 & 0.4855 & 0.6025 & 0.5779 
& 0.5300 & 0.5580 & 0.5815 & 0.4493 & 0.5708 & 0.5379 
"""

tmp_tp = """
& -0.0478  &  0.0102 &  0.0478 &  0.0499 & -0.0240 &  0.0072  
& -0.0102  &  0.0076 &  0.0010 & -0.0063 &  0.0161 &  0.0016 
&  0.0448  &  0.0348 &  0.0287 & -0.0177 & -0.0281 &  0.0125  
& -0.0139  & -0.0016 &  0.0016 & -0.0013 & -0.0006 & -0.0031 
"""

spa_tp = """
& 0.0336  & 0.3359 &  0.0270 & 0.2014 &  0.2709 &  0.1738
& 0.0532  & 0.1941 &  0.0442 & 0.0453 &  0.0204 &  0.0714 
& 0.1678  & 0.2776 &  0.0538 & 0.0299 &  0.3093 &  0.1510  
& 0.2175  & 0.2998 & -0.0039 & 0.1680 &  0.1945 &  0.1752
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


tmp = clean_data(tmp_tp)
spa = clean_data(spa_tp)

print_latex_table(tmp, spa)

