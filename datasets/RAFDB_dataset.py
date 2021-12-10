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
    0: "surprise",
    1: "fear",
    2: "disgust",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral",
}

BASE_EMOTION_DICT_INVERSE = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}



class RAFDBDataset(BaseDataset):
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
        self._data = open(os.path.join(self.dataset_path, 'labels.txt'), 'r').read()
        self._data = self._data.split('\n')
        self._data = [d for d in self._data if self._stage in d]

        self.images = []
        self.labels = []
        for d in self._data:
            try:
                file, label = d.split(' ')
            except:
                break
            if (self._stage in file):
                image = Image.open(os.path.join(self.dataset_path, 'Image/adapted', file.replace('.jpg', '_adapted.jpg'))).convert('RGBA')
                self.images.append(ImageOps.grayscale(image))
                self.labels.append(BASE_EMOTION_DICT_INVERSE[EMOTION_DICT[int(label)-1]])

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        image = np.asarray(self.images[index])
        image = image.astype(np.uint8)

        # print(self._image_size)
        image = cv2.resize(image, self._image_size)

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        if self.affine:
            image = seg(image=image)

        # if self._stage == "test" and self._tta == True:
        #     images = [seg(image=image) for i in range(self._tta_size)]
        #     # images = [image for i in range(self._tta_size)]
        #     images = list(map(self._transform, images))
        #     target = self._emotions.iloc[idx].idxmax()
        #     return images, target

        image = self._transform(image)
        target = self.labels[index]
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.labels)

    def get_emotion(self, index):
        if (index in EMOTION_DICT):
            return EMOTION_DICT[index]
        else:
            return "error"
