import os
from tqdm import tqdm
import torch
import numpy as np
from torchvggish import vggish, vggish_input
from dpcv.modeling.networks.multi_modal_pred_net import resnet101_visual_feature_extractor
from dpcv.data.datasets.ture_personality_data import Chalearn21FrameData, Chalearn21AudioDataPath


class TPExtractVisualFeatureData:
    def __init__(self, data_root, data_type, task, trans=None, save_to="", sample_num=2000):
        assert data_type in ["frame", "face", "audio"], "data_type should be one of [frame, face or video]"

        self.type = data_type
        if data_type == "audio":
            self.dataset = {
                "train": Chalearn21AudioDataPath(data_root, "train", task),
                "valid": Chalearn21AudioDataPath(data_root, "valid", task),
                "test": Chalearn21AudioDataPath(data_root, "test", task),
            }
        else:
            self.dataset = {
                "train": Chalearn21FrameData(
                    data_root, "train", task, data_type, even_downsample=sample_num, trans=trans, segment=False),
                "valid": Chalearn21FrameData(
                    data_root, "valid", task, data_type, even_downsample=sample_num, trans=trans, segment=False),
                "test": Chalearn21FrameData(
                    data_root, "test", task, data_type, even_downsample=sample_num, trans=trans, segment=False),
            }
        self.model = self.get_extract_model()
        # os.makedirs(save_to, exist_ok=True)
        self.save_to = save_to

    def get_extract_model(self):
        if not self.type == "audio":
            model = resnet101_visual_feature_extractor()
        else:
            model = vggish()
        model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model.eval()

    def extract_and_save_feat(self):
        with torch.no_grad():
            # if not self.type == "audio":
            for split in ["train", "valid", "test"]:
                dataset = self.dataset[split]
                save_dir = os.path.join(self.save_to, f"{split}_{self.type}")
                os.makedirs(save_dir, exist_ok=True)
                for i in tqdm(range(len(dataset))):

                    if not self.type == "audio":
                        data, label = dataset[i]["image"], dataset[i]["label"]
                        file_path = dataset.get_file_path(i)
                        file_name = os.path.basename(file_path).replace(".jpg", "")
                        dir_path = os.path.dirname(file_path)
                        dir_name = "_".join(dir_path.split("/")[-2:])
                        feat = self.model(
                            data[None].to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        )
                        sample = {
                            "feature": feat,
                            "label": np.array(label, dtype="float32"),
                        }
                        save_to_dir = os.path.join(save_dir, dir_name)
                        os.makedirs(save_to_dir, exist_ok=True)
                        saved_path = os.path.join(save_to_dir, f"{file_name}.pkl")
                        torch.save(sample, saved_path)
                    else:
                        file_path, label = dataset[i]["aud_path"], dataset[i]["aud_label"]
                        dir_name = "_".join(file_path.split("/")[-2:]).replace(".wav", "")
                        example = vggish_input.wavfile_to_examples(file_path)
                        feat = self.model.forward(
                            example.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        )

                        sample = {
                            "feature": feat,
                            "label": np.array(label, dtype="float32"),
                        }
                        saved_path = os.path.join(save_dir, f"{dir_name}.pkl")
                        torch.save(sample, saved_path)


if __name__ == "__main__":
    from dpcv.data.transforms.transform import set_transform_op

    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")
    task = "talk"
    type = "frame"

    transform = set_transform_op()
    extractor = TPExtractVisualFeatureData(
        data_root="datasets/chalearn2021",
        data_type=type,
        task=task,
        trans=transform,
        save_to=f"datasets/extracted_feature_tp/{task}",
    )
    extractor.extract_and_save_feat()
