import random

import torch
import numpy as np
import pandas as pd
from fastai.vision.all import aug_transforms, ImageDataLoaders, Resize


class DataAugmentor:
    SEED = 5242
    TRAIN_FILE_NAME = "train_label.csv"
    TRAIN_IMAGES_FOLDER = "train_image"

    @staticmethod
    def _detect_device(seed_value):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(seed_value)  # cpu vars
        torch.manual_seed(seed_value)  # cpu  vars
        random.seed(seed_value)  # Python
        if device == "gpu":
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # gpu vars
            torch.backends.cudnn.deterministic = True  # needed
            torch.backends.cudnn.benchmark = False
        print(f"Using device {device} for training and testing..")
        return device

    @classmethod
    def fetch(cls, df, imagesize: int, batchsize: int):
        """
        Function to create a dataloader with the required transformations
        """
        device = cls._detect_device(cls.SEED)
        # We apply the following augmentations:
        # - Flip the images vertically
        # - Rotate the images randomly, with max rotation angle as 10 degres
        # - Zoom in and out of the image, with max zoom as 1.1x
        # - Change lighting of the image by 20%
        # - Add warping to the image
        # - Resize the image to given size. Typically, this is 224x224 and 320x320
        # - Perform affine transformations on the image with probability 0.75
        # - Perform lighing transformation on the image with probability 0.75
        trfm = aug_transforms(
            do_flip=True,
            flip_vert=True,
            max_rotate=10.0,
            max_zoom=1.1,
            max_lighting=0.2,
            max_warp=0.2,
            size=imagesize,
            p_affine=0.75,
            p_lighting=0.75,
        )
        image_loader = ImageDataLoaders.from_df(
            df[["name", "label"]],
            valid_pct=0.1,
            seed=cls.SEED,
            device=device,
            item_tfms=Resize(460),
            bs=batchsize,
            batch_tfms=trfm,
        )
        return image_loader

    @classmethod
    def create_dataframe(cls, train_path):
        df = pd.read_csv(
            f"{train_path}/{cls.TRAIN_FILE_NAME}", names=["image_id", "label"], header=0
        )
        df["name"] = df["image_id"].apply(
            lambda x: f"{train_path}/{cls.TRAIN_IMAGES_FOLDER}/{x}.png"
        )
        return df
