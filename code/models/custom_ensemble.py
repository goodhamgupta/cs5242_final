import os

from ..data import DataAugmentor
from fastai.vision import *
from fastai.vision.all import *


class CustomEnsemble:

    TOTAL_EPOCHS = 10
    FREEZE_EPOCHS = 10

    @classmethod
    def get_models(cls, train_df):
        # Empty List of ensemble model, We will store the trained learner objects here.
        ensemble_models = []
        model_list = [
            densenet169,
            densenet201,
            resnet101,
            resnet152,
        ]  # List of models for the enseble
        for i in range(len(model_list)):
            print(f"-----Training model: {i+1}--------")
            small_data = DataAugmentor.fetch(train_df, 224, 64)
            # Since the validation set is empty, accuracy and RocAuc will be None.
            # To enable these metrics, change the valid_pct in DataAugmentor class to 0.1 i.e 10% dataset as validation set. Line number 54.
            # Replace metrics in cnn_learner with the one in the line below
            # metrics=[error_rate, accuracy, RocAuc()],
            learn_resnet = cnn_learner(small_data, model_list[i], metrics=[error_rate])

            print("training for 224x224")
            learn_resnet.set_data = small_data  # Train the model for imagesize 224
            lr_find_result = learn_resnet.lr_find()
            # using the learning rate for the first model
            learn_resnet.fine_tune(
                cls.TOTAL_EPOCHS, lr_find_result.lr_min, freeze_epochs=cls.FREEZE_EPOCHS
            )

            ## Training for larger image size
            print("training for 320x320")
            large_data = DataAugmentor.fetch(train_df, 320, 64)
            learn_resnet.set_data = large_data
            learn_resnet.fine_tune(cls.TOTAL_EPOCHS, freeze_epochs=cls.FREEZE_EPOCHS)

            ensemble_models.append(learn_resnet)
            print(f"-----Training of model {i+1} complete----")
        return ensemble_models
