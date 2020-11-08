import os
import pandas as pd
from fastai.vision.data import get_image_files


class CustomEnsemblePredict:
    """
    Class to perform prediction of dataset given the final model.
    - No test time augmentation available
    - Each model makes a seperate prediction for each image.
    - We combine predictions using the mode i.e most common prediction for a given image
    - Write final output to csv
    """

    @staticmethod
    def _create_submission(all_predictions):
        final_result_df = pd.concat(all_preds, axis=1)
        pred_img_nums = final_result_df.iloc[:, 0].values
        # Combine predictions using mode i.e most common value for image
        pred_img_vals = final_result_df["Label"].mode(axis=1).values
        # Take first value only as the returned value is an array
        clean_vals = [int(x[0]) for x in pred_img_vals]
        final_df = pd.DataFrame({"ID": pred_img_nums, "Label": clean_vals})
        return final_df

    @classmethod
    def create_df(cls, learner, test_path, num_images):
        test_nums = [f"{test_path}/{num}.png" for num in range(num_images)]
        test_images = get_image_files(test_path)
        test_df = pd.DataFrame({"name": test_nums})
        predictions = []
        for timage in test_images:
            predictions.append(test_learner.predict(timage))
        fnames = [f"{fname.name.replace('.png', '')}" for fname in test_images]
        pred_labels = [int(record[0]) for record in predictions]
        pred_df = pd.DataFrame({"ID": fnames, "Label": pred_labels})
        return pred_df

    @classmethod
    def _fetch_num_images(cls, test_path):
        """
        Function to fetch the number of test images
        """
        return len(os.listdir(test_path))

    @classmethod
    def perform_prediction(cls, models, test_path):
        """
        Function to perform prediction
        """
        num_images = cls._fetch_num_images(test_path)
        all_predictions = []
        for model in models:
            pred_df = cls.create_df(model, test_path, num_images)
            all_predictions.append(pred_df)
        final_df = cls._create_submission(models)
        return final_df
