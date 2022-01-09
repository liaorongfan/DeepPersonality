import os
from PIL import Image


class VideoLoader:

    def __init__(self, image_name_formatter=lambda x: f"frame_{x}.jpg"):
        self.image_name_formatter = image_name_formatter

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_path, self.image_name_formatter(i))
            if os.path.exists(image_path):
                video.append(Image.open(image_path))
        return video

