from ..data import DataAugmentor
from fastai.vision import *
from fastai.vision.all import *


class CustomEnsemble:

    TOTAL_EPOCHS = 10
    FREEZE_EPOCHS = 3

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
            data = DataAugmentor.fetch(train_df, 224, 64)
            learn_resnet = cnn_learner(
                data,
                model_list[i],
                metrics=[error_rate, accuracy, RocAuc()],
                model_dir="/tmp/model/",  # Destination dir for model weights
            )

            #print("training for 224x224")
            #learn_resnet.set_data = (224, 4)  # Train the model for imagesize 224
            #lr_find_result = learn_resnet.lr_find()
            ## using the learning rate for the first model
            #learn_resnet.fine_tune(
            #    cls.TOTAL_EPOCHS, lr_find_result[0], freeze_epochs=cls.FREEZE_EPOCHS
            #)

            ## Training for larger image size
            #print("training for 320x320")
            #learn_resnet.set_data = train_data(train_df, 320, 2)
            #learn_resnet.fine_tune(cls.TOTAL_EPOCHS, freeze_epochs=cls.FREEZE_EPOCHS)

            ## Save model weights
            #learn_resnet.save(f"ensem_model_{i}.weights")
            ensemble_models.append(learn_resnet)
            print(f"-----Training of model {i+1} complete----")
        return ensemble_models
