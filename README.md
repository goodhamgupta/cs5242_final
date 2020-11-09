# CS5242: Neural Networks and Deep Learning
This is a single-class multi-classification task based on medical images. More specifically, the images are based on Lung tomography.


## Resources

Download and unzip data from [Kaggle](https://www.kaggle.com/c/nus-cs5242/data).

- If you have kaggle cli install(package available [here](https://github.com/Kaggle/kaggle-api)), you can run the following command:
```
kaggle competitions download -c nus-cs5242; unzip nus-cs5242.zip
```

- Refactor the data folders into the following structure:

```
train_data/
    train_label.csv
    train_images/

test_data/
    test_images/
```


## Setup

- Clone the project
- Install the requirements
```
pip install -r requirements.txt
```

## Run

- We use a final ensemble of models the generate the prediction results in `submission.csv`. 

- You can run the project using the command:

```
python run.py --train_data train_data --test_data test_data
```

- Ensure that there are no trailing slashes in the argument. Example: `--train_data train_data/` is incorrect. Use `--train_data train_data` instead.

## Note

- The results from the model will not always match our leaderboard results. This is because:
  - We have set all possible random seed during the training and test procedure. However, because of multiprocessing to do image augmentation/training and floating point differences, we are unable to get an exact match with the leaderboard submission. 
  - Removing multiprocessing might help get slightly more accurate results, however I’ve noticed that this is extremely slow, because we use an ensemble of 4 models(resnets and densenets). 

Overall, we have noticed the following variation:

Public Leaderboard: 0.986-0.993
Private Leaderboard: 0.965-0.972
