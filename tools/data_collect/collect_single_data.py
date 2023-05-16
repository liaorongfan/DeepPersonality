import glob

def colloect_single_traits(log_dir="results/single_traits/A"):
    log_files = glob.glob(f"{log_dir}/*/*/log.log")
    results = []
    for file in log_files:
        model_name = file.replace(log_dir, "").split("/")[1]
        with open(file, 'r') as fo:
            lines = fo.readlines()
            result = lines[-1].split("INFO - ")[-1].split("mean")[0]
            results.append((model_name, result))
    for item in results:
        print(item)


if __name__ == "__main__":
    path = "results/single_traits/O"
    colloect_single_traits(path)