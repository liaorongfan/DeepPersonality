import numpy as np


def parse_tp_results(file="tp_result/deep_bimodal_reg_tp.txt", output_file="tp_output_deep_bimodal.txt"):
    with open(file, 'r') as f:
        lines = f.readlines()

    def clean(lines, str1, str2=""):
        lines_new = [line.replace(str1, str2) for line in lines]
        return lines_new

    lines = clean(lines, "{", "")
    lines = clean(lines, "}", "")
    lines = clean(lines, "mean:", ",")
    lines = clean(lines, "'O':", "")
    lines = clean(lines, "'C':", "")
    lines = clean(lines, "'E':", "")
    lines = clean(lines, "'A':", "")
    lines = clean(lines, "'N':", "")

    with open(output_file, "w") as fw:
        for line in lines:
            line = line.replace(",", " & ")
            fw.writelines(line)

    bound = 0
    for i, line in enumerate(lines):
        if "Face" in line:
            bound = i
            break
    frame_data, face_data = lines[1: bound], lines[bound + 1:]
    # print()

    def collect_session_data(data_ls):
        session_data = {}
        for line in data_ls[1:]:
            try:
                key, val = line.split(":")
            except:
                print(line)
            session_data[key.strip()] = val.strip()
        return session_data

    frame_session_data = {
        "talk": collect_session_data(frame_data[: 5]),
        "animal": collect_session_data(frame_data[5: 10]),
        "ghost": collect_session_data(frame_data[10: 15]),
        "lego": collect_session_data(frame_data[15: 20])
    }
    try:
        face_session_data = {
            "talk": collect_session_data(face_data[: 5]),
            "animal": collect_session_data(face_data[5: 10]),
            "ghost": collect_session_data(face_data[10: 15]),
            "lego": collect_session_data(face_data[15: 20])
        }
    except:
        face_session_data = []

    def calculate_mean_value(session_data, metric="acc"):
        separate_ccc_data = []
        separate_data = []
        for key, val in session_data.items():
            separate_ccc_data.append(val["ccc"])
            separate_data.append(val[metric])
        # print()
        separate_ccc_data = [list(map(float, line.split(","))) for line in separate_ccc_data]
        separate_ccc_mean = np.array(separate_ccc_data).mean(axis=0).round(4).tolist()
        separate_ccc_mean = str(separate_ccc_mean).strip("[]").replace(",", " & ")

        separate_data = [list(map(float, line.split(","))) for line in separate_data]
        separate_mean = np.array(separate_data).mean(axis=0).round(4).tolist()
        separate_mean = str(separate_mean).strip("[]").replace(",", " & ")

        return separate_ccc_mean, separate_mean

    fra_mean_ccc, fra_mean_mse = calculate_mean_value(frame_session_data, "mse")
    if len(face_session_data) > 0:
        fac_mean_ccc, fac_mean_mse = calculate_mean_value(face_session_data, "mse")
    else:
        fac_mean_ccc, fac_mean_mse = 0, 0

    with open(output_file, 'a') as fa:

        frame_string = f"Frame: \n\t ccc: {fra_mean_ccc} \n\t mse: {fra_mean_mse}"
        fa.writelines(frame_string)
        face_string = f"\nFace: \n\t ccc: {fac_mean_ccc} \n\t mse: {fac_mean_mse}"
        fa.writelines(face_string)


if __name__ == "__main__":
    import os
    # files = [
    #     ("deep_bimodal_reg_tp.txt", "tp_output_deep_bimodal.txt"),
    #     ("interpre_img_tp.txt", "tp_output_interpret_img.txt"),
    #     ("audioviaual_resnet_tp.txt", "tp_output_audiovisual_resnet.txt"),
    #     ("bimodal_lstm.txt", "tp_output_bimodal_lstm.txt"),
    #     ("crnet_tp.txt", "tp_output_crnet.txt"),
    #     ("persemon.txt", "tp_output_persemon.txt"),
    #     ("aud_lstm_tp.txt", "tp_output_aud_lstm.txt"),
    #     ("aud_interpret_aud.txt", "tp_output_interpret_aud.txt"),
    # ]

    # dir = "tp_result_vl"
    # files = [
    #     ("3DResNet.txt", "tp_3DResNet_vl.txt"),
    #     ("slow_fast.txt", "tp_slow_fast_vl.txt"),
    #     ("tpn.txt", "tp_tpn_vl.txt"),
    #     ("vat.txt", "tp_vat_vl.txt")
    # ]

    # dir = "tp_result"
    # files = [
    #     ("multi_modal_pred.txt", "tp_multi_modal_pred.txt"),
    #     ("multi_modal_pred_audio.txt", "tp_multi_modal_pred_audio.txt"),
    #     ("multi_modal_aud_vis.txt", "tp_multi_modal_aud_vis.txt"),
    # ]

    dir = "visual_in_bimodal"
    files = [
        ("10_bimodel_reg_vis.txt", "10.txt"),
        ("11_bimodal_renet_18.txt", "11.txt"),
        ("12_crnet_vis.txt", "12.txt"),
        ("13_bimodal_lstm_vis.txt", "13.txt"),
    ]
    for sour, tgt in files:
        parse_tp_results(os.path.join(dir, sour), tgt)
