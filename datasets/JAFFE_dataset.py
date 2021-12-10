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
    0: "happy",
    1: "sad",
    2: "surprise",
    3: "angry",
    4: "disgust",
    5: "fear"
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



class JAFFEDataset(BaseDataset):
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
        self._data = pd.read_csv(os.path.join(self.dataset_path, 'annotations.csv'), delimiter=' ')
        self.image_names = os.listdir(os.path.join(self.dataset_path, 'adapted'))

        self.images = []
        self.labels = []
        for i in range(len(self._data)):
            cur = self._data.iloc[i]
            file = self.get_name_from_partial(cur['PIC'].replace('-', '.'))
            if (file != ''):
                image = Image.open(os.path.join(self.dataset_path, 'adapted', file)).convert('RGBA')
                label = np.argmax(cur[1:-1]) #will be in range 0-5

                self.images.append(ImageOps.grayscale(image))
                self.labels.append(BASE_EMOTION_DICT_INVERSE[EMOTION_DICT[int(label)]])

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def get_name_from_partial(self, partial):
        for i in self.image_names:
            if partial in i:
                return i
        return ''

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

    def get_emotion(self, index):
        if (index in EMOTION_DICT):
            return EMOTION_DICT[index]
        else:
            return "error"
