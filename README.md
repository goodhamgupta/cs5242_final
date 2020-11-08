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

## Run project

- We use a final ensemble of models the generate the prediction results in `submission.csv`. 

- You can run the project using the command:

```
python run.py train_data test_data
```
