import numpy as np
import scipy.stats as stats


aud_ip = """
& 0.1968  & 0.1497  & 0.1738  & 0.1295  & 0.1780   & 0.1655 
& -0.0004 & -0.0005 & 0.0004  & -0.0005 & -0.0008  & 0.0004 
& 0.1293  & 0.0830  & 0.0458  & 0.1101  & 0.1548   & 0.1046 
& 0.4122  & 0.3406  & 0.3846  & 0.2857  & 0.4306   & 0.3707
& 0.4516  & 0.4493  & 0.4429  & 0.3127  & 0.4500   &0.4213
"""

vis_ip = """
& 0.5923 & 0.6912 & 0.6436 & 0.5195 & 0.6273 & 0.6148
& 0.6216 & 0.6753 & 0.6836 & 0.5228 & 0.6456 & 0.6298
& 0.5882 & 0.6550 & 0.6326 & 0.5003 & 0.6199 & 0.5992
& 0.5858 & 0.6750 & 0.5997 & 0.4971 & 0.5765 & 0.5868 
& 0.5693 & 0.6254 & 0.6070 & 0.4855 & 0.6025 & 0.5779
"""

aud_vis_ip = """
& 0.4341  & 0.4645  & 0.4553  & 0.3519  & 0.4588  & 0.4329
& 0.0000  & 0.0000  & 0.0000  & 0.0000  & 0.0000  & 0.0000
& 0.4150  & 0.3671  & 0.3889  & 0.2679  & 0.4181  & 0.3714
& 0.5193  & 0.5106 & 0.5024   & 0.4026  & 0.5119  & 0.4894
& 0.5618  & 0.6421 & 0.5921   & 0.4620  & 0.5734  & 0.5663
"""





aud_tp = """
& 0.0002 & -0.0002 & 0.0002 & 0.0001 & 0.0001 & 0.0001 
& 0.0000 & -0.0001 & 0.0000 & 0.0000 & 0.0000 & 0.0000 
& -0.0459 & 0.1045 & -0.0416 & 0.0429 & 0.0015 & 0.0123
& 0.0005 & 0.0290 & 0.0201 & 0.0335 & 0.0267 & 0.0220 
& 0.0688 & 0.1882 & 0.1310 & 0.1069 & 0.0412 & 0.1072 
"""

vis_tp = """
&  0.0336  &  0.3359 &  0.0270 & 0.2014 &  0.2709 & 0.1738
&  0.1678   & 0.2776 & 0.0538 &  0.0299 & 0.3093 &  0.1510
&  0.2175  & 0.2998  & -0.0039 & 0.1680 &  0.1945 & 0.1752
&  0.0532   &  0.1941 &  0.0442 &  0.0453 &  0.0204 &  0.0714 
&  0.0648 &  0.2424 &  0.0349 &  0.0305 &  -0.0009 &  0.0743  
"""

aud_vis_tp = """
&  0.0008 &  0.1154 & -0.0132 & -0.0222 & 0.1219 & 0.0405
&  0.0001 & -0.0001 &  0.0001 & -0.0001 &  0.0005 &  0.0001 
& -0.0530 & 0.1290&  0.0230 &  0.0310 &  0.0002 &  0.0260 
& 0.0998 &  0.1780 & 0.1158 &  0.2168 &  0.0449&  0.1311
& -0.0348 &  0.0468 & 0.0302 & 0.0397&  0.0041 &  0.0171 
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


def transform(aud, vis, aud_vis):
    aud = clean_data(aud)
    vis = clean_data(vis)
    aud_vis = clean_data(aud_vis)
    return aud, vis, aud_vis


aud, vis, aud_vis = transform(aud_ip, vis_ip, aud_vis_ip)
# compute_p_value(aud_tp, vis_tp)

print_latex_table(aud_vis, vis)
print()
print_latex_table(aud, vis)
print()
print_latex_table(aud, aud_vis)
