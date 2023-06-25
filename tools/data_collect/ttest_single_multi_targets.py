import numpy as np
import scipy.stats as stats


single_ip = """
& 0.3846  &  0.1504  &  0.4136    &  0.2041    &  0.4177   & 0.3141
& 0.4441  &  0.4563  &  0.4384    &  0.3097    &  0.4389   & 0.4175
& 0.2332  & 0.3223   & 0.2589     &  0.0806    &  0.2230  & 0.2236
& 0.1391  & 0.1636   & 0.1549     & 0.0526     &  0.1321  & 0.1284
& 0.3991  & 0.4243   & 0.3952     & 0.2018     & 0.4298  & 0.3700
& 0.3968  & 0.5318   & 0.4667     & 0.2821     & 0.3814  & 0.4118
"""

multi_ip = """
& 0.4122  & 0.3406  & 0.3846  & 0.2857  & 0.4306   & 0.3707
& 0.4516  & 0.4493  & 0.4429  & 0.3127  & 0.4500   & 0.4213
& 0.5923 & 0.6912 & 0.6436 & 0.5195 & 0.6273 & 0.6148
& 0.6216 & 0.6753 & 0.6836   & 0.5228  & 0.6456 & 0.6298
& 0.5193  & 0.5106  & 0.5024  & 0.4026  & 0.5119  & 0.4894
& 0.5618  & 0.6421  & 0.5921  & 0.4620  & 0.5734  & 0.5663 
"""

single_tp = """
& 0.0031  &  0.0121  &  0.0058  &  -0.0280  &  0.0007  & -0.0013 
& -0.0062  &  0.1473  &   0.0669  &  0.1422   &   0.1003  &  0.0900
& 0.1450 & 0.3307   & -0.0007    &  -0.0112  &  0.2294   & 0.1386 
& 0.0912  & 0.0277   & 0.0571    & 0.0009    &  0.0032   & 0.0360
& 0.0000  & 0.0000   & 0.0001    & 0.3527    & 0.0000   & 0.0705
& 0.0311  & 0.0371   & 0.0068    & 0.0837    & 0.0247   & 0.0367
"""

multi_tp = """
& 0.0005  & 0.0290 & 0.0201  & 0.0335  & 0.0267   & 0.0220
& 0.0688  & 0.1882 & 0.1310  & 0.1069  & 0.0412   & 0.1072 
&  0.2175  &  0.2998 & -0.0039 &  0.1680 &  0.1945 &  0.1752
& -0.0139  & -0.0016 &  0.0016 & -0.0013 & -0.0006 & -0.0031
& 0.0998 &  0.1780 &  0.1158 &  0.2168 &  0.0449 &  0.1311
& -0.0348 &  0.0468 &  0.0302 &  0.0397 &  0.0041 &  0.0171
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


single_ip = clean_data(single_ip)
multi_ip = clean_data(multi_ip)
print_latex_table(single_ip, multi_ip)
print()
single_tp = clean_data(single_tp)
multi_tp = clean_data(multi_tp)
print_latex_table(single_tp, multi_tp)
