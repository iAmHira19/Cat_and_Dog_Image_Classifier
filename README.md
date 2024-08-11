# Cat and Dog Image Classifier

## Overview

This project involves building a convolutional neural network (CNN) to classify images of cats and dogs using TensorFlow 2.0 and Keras. The goal is to develop a model that can accurately distinguish between cats and dogs with at least 63% accuracy, and ideally reach 70% accuracy.

## Dataset

The dataset is organized into three directories:

```
cats_and_dogs
|__ train:
|______ cats: [cat.0.jpg, cat.1.jpg ...]
|______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
|______ cats: [cat.2000.jpg, cat.2001.jpg ...]
|______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```

## Dependencies

To run this project, you'll need the following Python libraries:

- `tensorflow==2.0`
- `keras`
- `numpy`
- `matplotlib`

You can install these dependencies using pip:

```bash
pip install tensorflow==2.0 keras numpy matplotlib
```

## Instructions

### 1. Set Dataset Paths

Update the dataset paths in the code to point to the directories where your dataset is stored. The dataset directories should include the training, validation, and test images.

```python
TRAINING_DIR = '/path/to/train'
VALIDATION_DIR = '/path/to/validation'
TEST_DIR = '/path/to/test'
```

### 2. Create Image Data Generators

Use `ImageDataGenerator` to preprocess the images. This includes rescaling pixel values to the range [0, 1] and applying data augmentation techniques for the training dataset.

- **Training Data Generator**: Applies random transformations like rotation, shifting, and flipping to augment the training data.

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, fill_mode='nearest')
train_data_gen = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')
```

- **Validation Data Generator**: Only rescales pixel values without additional transformations.

```python
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data_gen = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary')
```

- **Test Data Generator**: Similar to validation data generator but without shuffling to maintain the order of predictions.

```python
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_gen = test_datagen.flow_from_directory(directory=TEST_DIR,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False)
```

### 3. Build and Compile the CNN Model

Create a CNN model using Keras with several convolutional layers, max pooling layers, and dense layers. Compile the model using the Adam optimizer and binary cross-entropy loss function.

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 4. Train the Model

Train the model using the training and validation datasets. Monitor the accuracy and loss during training.

```python
history = model.fit(train_data_gen,
                    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
                    epochs=10,
                    validation_data=validation_data_gen,
                    validation_steps=validation_data_gen.samples // BATCH_SIZE)
```

### 5. Evaluate the Model

Evaluate the trained model on the test dataset and print the test accuracy.

```python
test_loss, test_acc = model.evaluate(test_data_gen)
print(f'Test accuracy: {test_acc}')
```

### 6. Visualize Performance

Plot training and validation accuracy over epochs to visualize model performance.

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and Keras for providing tools to build and train deep learning models.
- The dataset provider for making the dataset available for use in this project.
```

