import torch
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
from torchvggish import vggish, vggish_input
from dpcv.data.transforms.transform import set_transform_op
from dpcv.data.datasets.bi_modal_data import VideoData
from dpcv.modeling.networks.multi_modal_pred_net import resnet101_visual_feature_extractor
import pickle


class ExtractVisualFeatureData(VideoData):
    def __init__(self, data_root, img_dir, label_file, save_to, length=100, suffix="frame_"):
        super().__init__(data_root, img_dir, label_file)
        self.len = length
        self.trans = set_transform_op()
        self.model = self.get_extract_model()
        os.makedirs(save_to, exist_ok=True)
        self.save_to = save_to
        self.suffix = suffix

    def get_sample_frames(self, idx):
        img_dir = self.img_dir_ls[idx]
        img_path_ls = glob.glob(f"{img_dir}/*.jpg")
        img_path_ls = sorted(
            img_path_ls,
            key=lambda x: int(
                str(Path(x).stem).replace(f"{self.suffix}", "")
            )
        )
        sample_frames_id = np.linspace(
            0, len(img_path_ls), self.len, endpoint=False, dtype=np.int16
        ).tolist()
        img_path_ls_sampled = [img_path_ls[idx] for idx in sample_frames_id]
        img_obj_ls = [Image.open(img_path) for img_path in img_path_ls_sampled]
        video_name = os.path.basename(img_dir)
        return video_name, img_obj_ls

    @staticmethod
    def get_extract_model():
        model = resnet101_visual_feature_extractor()
        return model.eval()

    def extract_and_save_feat(self):
        with torch.no_grad():
            for i in tqdm(range(len(self.img_dir_ls))):
                video_name, img_obj_ls = self.get_sample_frames(i)
                img_obj_tensor = self.img_transform(img_obj_ls)
                label = self.get_ocean_label(i)
                feat = self.model(img_obj_tensor)
                sample = {
                    "feature": feat,
                    "label": np.array(label, dtype="float32"),
                }
                saved_path = os.path.join(self.save_to, f"{video_name}.pkl")
                torch.save(sample, saved_path)

    def img_transform(self, img_obj_ls):
        img_obj_ls = [self.trans(img) for img in img_obj_ls]
        img_obj_ls = torch.stack(img_obj_ls, dim=0)
        img_obj_ls = img_obj_ls.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return img_obj_ls


class ExtractAudioFeatureData:
    def __init__(self, aud_dir, anno_path, save_to):
        self.aud_files = glob.glob(f"{aud_dir}/*.wav")
        self.model = self.get_extract_model()
        self.annotation = self.get_annotation(anno_path)
        os.makedirs(save_to, exist_ok=True)
        self.save_to = save_to

    @staticmethod
    def get_extract_model():
        # Initialise model and download weights
        embedding_model = vggish()
        embedding_model.eval()
        return embedding_model

    def extract_and_save_feat(self):
        with torch.no_grad():
            for file in tqdm(self.aud_files):
                file_name = Path(file).stem
                label = self.get_ocean_label(file_name)
                example = vggish_input.wavfile_to_examples(file)
                embeddings = self.model.forward(example)
                saved_path = os.path.join(self.save_to, f"{file_name}.pkl")
                sample = {"feature": embeddings, "label": np.array(label, dtype="float32")}
                torch.save(sample, saved_path)

    def get_ocean_label(self, file_name):
        video_name = f"{file_name}.mp4"
        score = [
            self.annotation["openness"][video_name],
            self.annotation["conscientiousness"][video_name],
            self.annotation["extraversion"][video_name],
            self.annotation["agreeableness"][video_name],
            self.annotation["neuroticism"][video_name],
        ]
        return score

    @staticmethod
    def get_annotation(label_path):
        with open(label_path, "rb") as f:
            annotation = pickle.load(f, encoding="latin1")
        return annotation


if __name__ == "__main__":
    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")
    extractor = ExtractVisualFeatureData(
        data_root="datasets",
        img_dir="image_data/test_data_face",
        label_file="annotation/annotation_test.pkl",
        save_to="datasets/extracted_feature_impression/test_face",
        suffix="face_",
    )
    extractor.extract_and_save_feat()


    # extr = ExtractAudioFeatureData(
    #     aud_dir="datasets/voice_data/voice_raw/validationData",
    #     anno_path="datasets/annotation/annotation_validation.pkl",
    #     save_to="datasets/extracted_feature_impression/valid_aud"
    # )
    # extr.extract_and_save_feat()
