import argparse

parser = argparse.ArgumentParser(description="Create model and perform prediction.")
parser.add_argument(
    "--train_data",
    type=str,
    help="path to the folder containing training dataset and train_label.csv",
)
parser.add_argument("--test_data", help="path to the folder containing the test images")

args = parser.parse_args()
print(args)
