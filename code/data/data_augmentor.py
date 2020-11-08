class DataAugmentor:

    @classmethod
    def fetch(cls, df, imagesize: int, batch_size: int):
        """
        Function to create a dataloader with the required transformations
        """
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
