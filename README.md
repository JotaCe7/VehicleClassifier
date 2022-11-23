# Vehicle classification from images

## 1. Install

You can use `Docker` to install all the needed packages and libraries easily. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t vehicleclassification_jc --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t vehicleclassification_jc --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile_gpu .
```

### Run Docker

- **CPU:**

```bash
$ docker run --rm --net host -it \
    -v "$(pwd):/home/app/src" \
    --workdir /home/app/src \
    vehicleclassification \
    bash
```

- **GPU:**

```bash
$ docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    vehicleclassification_gpu \
    bash
```


### Run Unit test


```bash
$ pytest tests/
```

### 2. Preparing data

Extract the images from the file `car_ims.tgz` and put them inside the `data/` folder. Also place the annotations file (`car_dataset_labels.csv`) in the same folder. It should look like this:

```
data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
```

Then, you should be able to run the script `scripts/prepare_train_test_dataset.py`. It will format your data in a way Keras can use for training our CNN model.

You will have to complete the missing code in this script to make it work.

### 3. Train CNN model (Resnet50)

After we have our images in place, it's time to create our first CNN and train it on our dataset. To do so, we will make use of `scripts/train.py`.

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.

To train a new model, create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. We also encourage you to store the model weights and training logs inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```

### 4. Evaluate your trained model

Use the notebook `notebooks/Model Evaluation.ipynb` to do evaluate your model.

### 5. Improve classification by removing noisy background

As we can see in the `notebooks/EDA.ipynb` file. Most of the images have a background that may affect our model learning during the training process.

It's a good idea to remove this background. One thing we can do is to use a Vehicle detector to isolate the car from the rest of the content in the picture.

We use [Detectron2](https://github.com/facebookresearch/detectron2) framework for this. It offers a lot of different models, you can check in its [Model ZOO](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn).

In particular, we use a detector model trained on [COCO](https://cocodataset.org) dataset which has a good balance between accuracy and speed. This model can detect up to 80 different types of objects but here we're only interested on getting two out of those 80, those are the classes "car" and "truck".