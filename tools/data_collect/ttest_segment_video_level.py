import numpy as np
import scipy.stats as stats

seg_ip = """
& 0.3248 & 0.3601 & 0.3601 & 0.2120 & 0.3352 & 0.3185 
& 0.0256 & 0.0320 & 0.0185 & 0.0105 & 0.0184 & 0.0210   
& 0.4427 & 0.4767 & 0.4998 & 0.3230 & 0.4675 & 0.4420 
& 0.6216 & 0.6753 & 0.6836 & 0.5228 & 0.6456 & 0.6298
"""

vid_ip = """
& 0.1171 & 0.1207  & 0.1199 & 0.0387  & 0.0954 & 0.0983  
& 0.0318 & 0.0463  & 0.0255 & 0.0176  & 0.0297 & 0.0302  
& 0.0709 & -0.0188 & 0.0166 & -0.0007 & 0.0654 & 0.0267 
& 0.1525 & 0.1794  & 0.1727 & 0.0783  & 0.1457 & 0.1457 
"""

seg_tp = """
& -0.0478  &  0.0102 &  0.0478 &  0.0499 & -0.0240 &  0.0072  
& -0.0102  &  0.0076 &  0.0010 & -0.0063 &  0.0161 &  0.0016 
&  0.0448  &  0.0348 &  0.0287 & -0.0177 & -0.0281 &  0.0125  
& -0.0139  & -0.0016 &  0.0016 & -0.0013 & -0.0006 & -0.0031 
"""

vid_tp = """
& 0.0468  &   0.0020   &   0.4706  &   0.089   &  -0.1329  &  0.0951  
& -0.0077 &   0.0150   &   0.0246  &   -0.0043 &  -0.0042  &  0.0047 
& -0.1159 &   -0.1285  &   0.1089  &   -0.1593 &  -0.0557  &  -0.0701  
& -0.0031 &   0.0073   &   -0.0082 &   0.0000  &   0.0019  &  -0.0004 
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


tmp = clean_data(seg_ip)
spa = clean_data(vid_ip)
print_latex_table(tmp, spa)

tmp = clean_data(seg_tp)
spa = clean_data(vid_tp)
print_latex_table(tmp, spa)
