import argparse

from code.data import DataAugmentor
from code.models import CustomEnsemble
from code.prediction import CustomEnsemblePredict

parser = argparse.ArgumentParser(description="Create model and perform prediction.")
parser.add_argument(
    "--train_data",
    type=str,
    help="path to the folder containing training dataset and train_label.csv",
)
parser.add_argument("--test_data", help="path to the folder containing the test images")

args = parser.parse_args()

train_df = DataAugmentor.create_dataframe(args.train_path)
ensemble_models = CustomEnsemble.get_models(train_df)
final_df = CustomEnsemblePredict.perform_prediction(ensemble_models, args.test_path)
final_df.to_csv("submission.csv", index=False)

