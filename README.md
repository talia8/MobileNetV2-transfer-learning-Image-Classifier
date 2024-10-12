# Cat vs Dog Image Classifier

## Project Overview
This project is a **Deep Learning Image Classifier** that can distinguish between images of **cats** and **dogs**. It utilizes **Transfer Learning** with the **MobileNetV2** architecture, pre-trained on the **ImageNet** dataset, to achieve high accuracy in classifying images. The classifier is built using **TensorFlow** and **Keras**, and trained on the **Cats vs Dogs** dataset provided by **TensorFlow Datasets (TFDS)**. The project includes preprocessing, training, evaluation, saving, and loading the model for future predictions.

## Features
- **Transfer Learning**: Uses the MobileNetV2 architecture with pre-trained ImageNet weights.
- **Data Augmentation**: Efficient preprocessing of the dataset with resizing and normalization.
- **Binary Classification**: Predicts whether an image is a cat or a dog.
- **Model Saving**: Saves the trained model for future use and prediction.
- **Prediction Script**: Load the saved model and classify new images.
  
## Dataset
The model is trained on the **Cats vs Dogs** dataset available from TensorFlow Datasets (TFDS). The dataset is split into:
- 80% for training
- 10% for validation
- 10% for testing

## How It Works

1. **Data Loading**: The dataset is loaded and split into training, validation, and test sets using `tfds.load()`.
2. **Preprocessing**: Each image is resized to 160x160 pixels, normalized, and shuffled for efficient training.
3. **Transfer Learning**: The MobileNetV2 model is used as a base model, excluding its top layers. A custom classification layer is added to predict cat vs dog.
4. **Training**: The model is trained on the dataset using **Binary Crossentropy Loss** and **RMSprop optimizer**.
5. **Evaluation**: The model's performance is evaluated on the validation dataset before and after training.
6. **Prediction**: A saved model can be loaded to classify new images.

## Model Training

The model is built using the following layers:
- **Base Model**: MobileNetV2 (pre-trained on ImageNet, with top layers removed).
- **Global Average Pooling Layer**: To reduce the feature map dimensions.
- **Dense Layer**: To classify images into either "cat" or "dog".

## How to Run the Project

1. **Download the Dataset**: TensorFlow will automatically download the Cats vs Dogs dataset when running the script.
2. **Train the Model**: Run the training script to train the model on the dataset.
3. **Save the Model**: After training, the model will be saved as `dogs_vs_cats.h5`.
4. **Test on New Images**: Load the saved model and use it to classify new images.

## Folder Structure

```
├── model.py                # Code for building and training the model
├── main.py              # Code for loading the model and predicting new images
└── README.md               # Project overview (this file)
```

## License

This project is open-source and free to use.
