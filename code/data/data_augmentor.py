class DataAugmentor:
    TRAIN_FILE_NAME = "train_labels.csv"

    @classmethod
    def fetch(cls, df, imagesize: int, batch_size: int):
        """
        Function to create a dataloader with the required transformations
        """
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
        img_loader = ImageDataLoaders.from_df(
            df[["name", "label"]],
            valid_pct=0.1,
            seed=5242,
            device="cuda:0",
            item_tfms=Resize(460),
            batch_tfms=trfm,
        )
        return image_loader

    @classmethod
    def create_dataframe(cls, train_path):
        df = pd.read_csv(
            f"{train_path}/cls.TRAIN_FILE_NAME", names=["image_id", "label"], header=0
        )
        return df
