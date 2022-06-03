# string_ls = [
# "{'O': 0.8976, 'C': 0.6484, 'E': 1.3352, 'A': 0.9222, 'N': 1.1521} mean: 0.9911",
# "{'O': -0.1188, 'C': 0.1426, 'E': -0.2425, 'A': 0.0121, 'N': -0.0582} mean: -0.053",
# "{'O': -0.0002, 'C': 0.0004, 'E': -0.0004, 'A': 0.0, 'N': -0.0002} mean: -0.0001",
# ]
#
# for string in string_ls:
#     # string = "{'O': -0.0799, 'C': 0.3223, 'E': 0.0984, 'A': 0.1041, 'N': -0.1377} mean: 0.0614"
#     str1, str2 = string.split("mean:")
#     dict = eval(str1)
#     for k, v in dict.items():
#         print(str(v) + " &", end=" ")
#     print(str2)


from collections import defaultdict
import numpy as np


model_dict = defaultdict(list)


with open("audio_statis_mse.txt", 'r') as f:
    lines = [line.strip().split("&") for line in f.readlines() if ":" not in line]

for line in lines:
    key = line[0].strip()
    value = [float(item.strip()) for item in line[1:]]
    model_dict[key].append(value)

for k, v in model_dict.items():
    v_arr = np.array(v)
    mean = v_arr.mean(axis=0).tolist()
    mean = [np.round(item, 4) for item in mean]
    mean.insert(0, k)
    for item in mean:
        print(" & " + str(item), end=" ")
    print(" \\\\")
print()


