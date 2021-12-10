import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
from PIL import Image, ImageOps


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class TFDDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]
        self.affine = configuration["affine"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = configuration["dataset_path"]
        self.img_dirs = os.listdir(self.dataset_path)

        self.images = []
        for d in self.img_dirs:
            cur_dir = os.path.join(self.dataset_path, d, 'adapted') 
            for i in os.listdir(cur_dir):
                img = Image.open(os.path.join(cur_dir, i)).convert('RGBA')
                img = ImageOps.grayscale(img)
                self.images.append(img)
        self.labels = [0 for i in range(len(self.images))]

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        image = np.asarray(self.images[index])#.reshape(48, 48)
        image = image.astype(np.uint8)

        # print(self._image_size)
        image = cv2.resize(image, self._image_size)

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        # if self._stage == "train":
        if self.affine:
            image = seg(image=image)

        image = self._transform(image)
        target = 0
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.images)

