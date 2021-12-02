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


class RAFDBDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

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
            image = Image.open(os.path.join(self.dataset_path, 'Image/adapted', file.replace('.jpg', '_adapted.jpg'))).convert('RGBA')
            self.images.append(ImageOps.grayscale(image))
            self.labels.append(int(label)-1)

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

        # if self._stage == "train":
        #     image = seg(image=image)

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
