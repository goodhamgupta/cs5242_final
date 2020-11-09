import argparse
import sys

from code.data import DataAugmentor
from code.models import CustomEnsemble
from code.prediction import CustomEnsemblePredict


class CreateSubmission:
    @classmethod
    def execute(cls, train_path, test_path):
        """
        Function to trigger the workflow.
        - Create augmented image dataset via DataAugmentor
        - Train ensemble of resnet and densenet models using CustomEnsemble
        - Create final prediction dataframe using CustomEnsemblePredict
        - Write dataframe to csv using pandas
        """
        train_df = DataAugmentor.create_dataframe(train_path)
        ensemble_models = CustomEnsemble.get_models(train_df)
        final_df = CustomEnsemblePredict.perform_prediction(ensemble_models, test_path)
        final_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    CreateSubmission.execute(sys.argv[1], sys.argv[2])
