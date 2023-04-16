import pickle
import itertools

with open("option_1_ls.pkl", "rb") as f:
    option_1 = pickle.load(f)
    print(option_1)

with open("option_2_ls.pkl", "rb") as f:
    option_2 = pickle.load(f)
    print(option_2)

with open("option_3_ls.pkl", "rb") as f:
    option_3 = pickle.load(f)
    print(option_3)
options = option_1 + option_2 + option_3
options = sorted(options, key=len)
print()
fold_ls = [[] for _ in range(10)]
index = [i for i in range(10)]

fold_ls[0] = options[29]
fold_ls[1] = options[28] + options[11]
fold_ls[2] = options[27] + options[12]
fold_ls[3] = options[26] + options[10] + options[9]
fold_ls[4] = options[25] + options[20]
fold_ls[5] = options[24] + options[21]
fold_ls[6] = options[23] + options[22]

fold_ls[7] = options[19] + options[18] + options[17]
fold_ls[8] = options[16] + options[15] + options[14] + options[8] + options[7]
fold_ls[9] = options[13] + options[6] + options[5] + options[4] + options[3] + options[2] + options[1] + options[0]


for s, e in itertools.permutations(index, 2):
    print("checking ...")
    du = list(set(fold_ls[s]) & set(fold_ls[e]))
    if len(du) > 0:
        print(du, fold_ls[s], fold_ls[e])
print("no duplications")

with open("10_fold_video_id_ls.txt", 'w') as f:
    for line in fold_ls:
        f.writelines(str(line) + "\n")


