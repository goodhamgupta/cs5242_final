from ..data import DataAugmentor
from fastai.all import *


class CustomEnsemble:

    TOTAL_EPOCHS = 10
    FREEZE_EPOCHS = 3

    @classmethod
    def get_models(cls):
        ensemble_models = [] # Empty List of ensemble model, We will store the trained learner object here.
        model_list = [densenet169, densenet201, resnet101,resnet152] # List of models for the enseble
        for i in range(len(model_list)):
            print(f'-----Training model: {i+1}--------')
            data = DataAugmentor.fetch(224,64)
            learn_resnet = cnn_learner(data, model_list[i], metrics=[error_rate, accuracy,RocAuc()],
                                       model_dir="/tmp/model/")

            print('training for 224x224')
            learn_resnet.set_data = (224,64) # Train the model for imagesize 128
            lr_find_result = learn_resnet.lr_find()
            # learn_resnet.recorder.plot(suggestion=True)
            learn_resnet.fine_tune(cls.TOTAL_EPOCHS, lr_find_result[0], freeze_epochs=cls.FREEZE_EPOCHS) # using the learning rate for the first model

            print('training for 320x320')
            learn_resnet.set_data = train_data(320,32) #Train the model for imagesize 150
            learn_resnet.fine_tune(cls.TOTAL_EPOCHS,freeze_epochs=cls.FREEZE_EPOCHS)   # using the learning rate assigned for the first model

            learn_resnet.save(f'ensem_model_{i}.weights')
            ens_model.append(learn_resnet)
            print(f'-----Training of model {i+1} complete----')
        return ens_model
