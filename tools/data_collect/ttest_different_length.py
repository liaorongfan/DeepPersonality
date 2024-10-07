import numpy as np
import scipy.stats as stats

f16_ip = """
& 0.3031 & 0.329 & 0.3185 & 0.1871 & 0.2725 & 0.2820
& 0.1680 & 0.1909 & 0.1572 & 0.0552 & 0.1447 & 0.1432
& 0.2293 & 0.2826 & 0.2671 & 0.1415 & 0.2397 & 0.2320
& 0.3251 & 0.3102 & 0.3427 & 0.1777 & 0.3151 & 0.2942
& 0.2202 & 0.1071 & 0.218 & 0.1000 & 0.1939 & 0.1678
& 0.1026 & 0.0703 & 0.073 & 0.0565 & 0.1211 & 0.0847 
"""

f32_ip = """
& 0.3248 & 0.3601 & 0.3601 & 0.2120 & 0.3352 & 0.3185
& 0.4427 & 0.4767 & 0.4998 & 0.3230 & 0.4675 & 0.442
& 0.6216 & 0.6753 & 0.6836 & 0.5228 & 0.6456 & 0.6298
& 0.3783 & 0.3547 & 0.3894 & 0.2189 & 0.356  & 0.3395 
& 0.2449  & 0.132   & 0.2513  & 0.1076  & 0.2226  & 0.1917
& 0.1504 & 0.0831 & 0.1145 & 0.07 & 0.1562 & 0.1148
"""

f64_ip = """
& 0.1123 & 0.1333 & 0.1234 & 0.0451 & 0.1065 & 0.1041
&  0.0079 & -0.0304 & 0.0079 & -0.0116 & -0.0172 & -0.0087
& 0.5598  & 0.5819 & 0.6413  & 0.4728 & 0.5891 & 0.5690
& 0.3947 & 0.3846 & 0.4035 & 0.2577 & 0.3947 & 0.367 
&  0.3136 & 0.2015 & 0.2998 & 0.1654 & 0.2832 & 0.2527 
& 0.157 & 0.0792 & 0.1382 & 0.0887 & 0.1514 & 0.1229 
"""

all_ip = """
& 0.0168 & 0.0298 & -0.0017 & -0.0090 & 0.0096 & 0.0091
& 0.4088 & 0.4280 &  0.4537 &  0.2936 & 0.4164 & 0.4001
& 0.5491 & 0.6109 &  0.6024 &  0.4301 & 0.5623 & 0.5510 
& 0.4516 & 0.4493 &  0.4429 &  0.3127 & 0.4500 & 0.4213
& 0.4122 & 0.3406 &  0.3846 &  0.2857 & 0.4306 & 0.3707
& 0.1570 & 0.0792 &  0.1382 &  0.0887 & 0.1514 & 0.1229
"""
# ----------------------------------------------------

f16_tp = """
& -0.1033 & -0.0316 & -0.0054 & -0.094 & 0.058 & -0.0352
&  -0.0136 & -0.049 & 0.0585 & -0.1084 & 0.0538 & -0.0117 
& -0.0018 & 0.0097 & 0.0035 & 0.004 & 0.0031 & 0.0037
& 0.0388 & 0.0847 & 0.1128 & 0.0218 & 0.0409 & 0.0598
& -0.0024 & -0.0218 & 0.004 & -0.008 & -0.0577 & -0.0172 
& -0.0011 & 0.0066 & -0.0003 & 0.0009 & 0.0005 & 0.0013
"""

f32_tp = """
& -0.0478  &  0.0102 &  0.0478 &  0.0499 & -0.0240 &  0.0072
& 0.0448  &  0.0348 &  0.0287 & -0.0177 & -0.0281 &  0.0125 
& -0.0139  & -0.0016 &  0.0016 & -0.0013 & -0.0006 & -0.0031
& -0.0502 & 0.0966 & 0.1028 & 0.1105 & 0.1505 & 0.0820
& -0.0017 & 0.005 & 0.0079 & 0.0024 & 0.014 & 0.0055
& -0.0216 & 0.0209 & 0.0065 & 0.0035 & -0.0001 & 0.0018
"""

f64_tp = """
&  0.0375 & 0.0137 & 0.212 & -0.1245 & 0.0400 &  0.0357
& -0.0007 & 0.0067 & -0.0328 & 0.0388 & 0.0078 & 0.004 
&  -0.0072 & 0.0142 & 0.0056 & 0.0254 & 0.0092 & 0.0095
& -0.0801 & 0.2605 & 0.0847 & 0.0438 & 0.1875 & 0.0993 
&  0.3042 & 0.1939 & 0.2952 & 0.1554 & 0.2823 & 0.2462
& -0.0075 & 0.0567 & 0.0048 & 0.0017 & -0.0002 & 0.0111 
"""

all_tp = """
& 0.0245 & -0.0164 & -0.0055 & -0.0204 &  0.0084 & -0.0019 
&-0.0494 &  0.0355 & -0.0028 & 0.0352 & 0.0125 & 0.0062 
& 0.0469 & -0.0483 & 0.0212 & -0.0682 & 0.0894 & 0.0082 
& 0.0688 & 0.1882 & 0.1310 & 0.1069 &  0.0412 & 0.1072
& 0.0005  &  0.0290  & 0.0201  & 0.0335  & 0.0267  & 0.0220
& -0.0459 & 0.1045  & -0.0416 & 0.0429  & 0.0015 & 0.0123 
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
    print()


def compute_2_clip(clip_1, clip_2):
    clip_1 = clean_data(clip_1)
    clip_2 = clean_data(clip_2)
    print_latex_table(clip_1, clip_2)


if __name__ == '__main__':
    compute_2_clip(f16_ip, f32_ip)
    compute_2_clip(f16_ip, f64_ip)
    compute_2_clip(f16_ip, all_ip)
    print("-------------------")
    compute_2_clip(f16_tp, f32_tp)
    compute_2_clip(f16_tp, f64_tp)
    compute_2_clip(f16_tp, all_tp)