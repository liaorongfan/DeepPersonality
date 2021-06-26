import subprocess
import os
import time


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        func(*args, **kwargs)
        time_end = time()
        time_spend = (time_end - time_start) / 60
        print('\n{0} cost time {1} min\n'.format(func.__name__, time_spend))

    return func_wrapper


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


@timer
def extract_face_feature(openface_exe_dir, data_root, dest_dir, missed_file=None):
    """
    Before execute that function please install openface
    https://github.com/TadasBaltrusaitis/OpenFace
    """
    dir_list = os.listdir(data_root)
    if missed_file:
        with open(missed_file, 'r') as f:
            dir_list = [file.strip() for file in f.readlines()]
    else:
        file_name = data_root.replace("/", "_") + ".txt"
        if not os.path.exists(file_name):
            with open(file_name, 'w') as fo:
                for item in dir_list:
                    fo.write(item + "\n")

    for i, dir in enumerate(dir_list):
        print(f"----------------------------> [{i}] processing images in [{dir}] <-------------------------------")
        dir_path = os.path.join(data_root, dir)
        mkdir_p(dest_dir)
        command = str(openface_exe_dir + "/FeatureExtraction" + " -fdir " + dir_path + " -rigid " + " -simalign "
                      + " -out_dir " + dest_dir + " -no2Dfp -no3Dfp -noMparams -noPose -noAUs -noGaze"
                      + "-q")
        print(command)
        # subprocess.call(command, shell=True)
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as err:
            print("ERROR: ", err)

        time.sleep(1)  # set time buffer for file io system


@timer
def check(path, check_file):
    dirs = os.listdir(path)
    dirs = list(set([item.replace("_aligned", "") for item in dirs if "aligned" in item]))
    print("processed num:", len(dirs))
    with open(check_file, 'r') as f:
        dir_ori = [item.strip() for item in f.readlines()]
        print("total num:", len(dir_ori))
    missed_file_list = []
    added_file_list = []
    for dir_processed in dirs:
        if not dir_processed in dir_ori:
            added_file_list.append(dir_processed)
    for dir in dir_ori:
        if not dir in dirs:
            missed_file_list.append(dir)

    if len(missed_file_list):
        print("there are some files missed")
        with open("missed_files.txt", 'w') as fo:
            for file in missed_file_list:
                fo.writelines(file + "\n")

    if len(added_file_list):
        print("there are some unexpected files added in")
        with open("added_files.txt", 'w') as fo:
            for file in added_file_list:
                fo.writelines(file + '/n')
    print("checked")


if __name__ == "__main__":
    extract_face_feature(
        "~/11-personality_traits/OpenFace/build/bin",
        "ImageData/trainingData",
        "../../../datasets/ImageData/trainingData_face",
        # "test_out",
    )
    # check("ImageData/trainingData_face", "ImageData_trainingData.txt")
