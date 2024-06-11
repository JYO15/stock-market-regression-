
# Data preprocessing
import pandas as pd
import numpy as np

# Visualization
from IPython.display import Image, display
import matplotlib.pyplot as plt

# Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers, Sequential

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from keras.preprocessing.image import ImageDataGenerator

# Suppressing warnings
from warnings import filterwarnings
filterwarnings('ignore')

display(Image(filename = "D:/archive/amer_sign2.png"))

# Reading data
train = pd.read_csv("D:/archive/sign_mnist_test/sign_mnist_test.csv")
test = pd.read_csv("D:/archive/sign_mnist_test/sign_mnist_test.csv")

# Separating images and labels
images, labels = train.iloc[:, 1:], train['label']
testImages, testLabels = test.iloc[:, 1:], test['label']

# Displaying 16 random images
def displayImg(images, title):
    plt.figure(figsize = (15, 10))
    
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.title(title[i])
        plt.imshow(np.reshape(images[i], (28, 28)), cmap = 'gray')
        plt.axis('off')
    plt.show()
    
rand = np.random.randint(0, images.shape[0] - 16)
displayImg(images.iloc[rand:rand + 16].values, labels.iloc[rand:rand + 16].values)


labels = to_categorical(labels, num_classes = 25)
testLabels = to_categorical(testLabels, num_classes = 25)

# Train-test validation split
trainX, testX, trainY, testY = train_test_split(images, labels, random_state = 0)

# Reshaping data and scaling it from 0 to 1
trainX, testX, testImages = [data.to_numpy().reshape(-1, 28, 28, 1) / 255 for data in [trainX, testX, testImages]]






model = Sequential([
    layers.Conv2D(64, (5, 5), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)),
    layers.Conv2D(64, (5, 5), activation = 'relu', padding = 'same'),
    layers.MaxPool2D(2),
    
    layers.Dropout(0.2),
    
    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    layers.MaxPool2D(2),
    
    layers.Dropout(0.2),
    layers.Flatten(),
    
    layers.Dense(256, activation = 'relu'),
    
    layers.Dropout(0.2),
    
    layers.Dense(25, activation = 'softmax')
])

# Defining optimizer, loss, and metrics
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Summary of model
model.summary()


# Adding callback to avoid overfitting
earlyStopping = EarlyStopping(
    monitor = 'val_accuracy',
    min_delta = 1e-4,
    patience = 5,
    restore_best_weights = True
)

# Adding a learning rate annealer
reduceLR = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience = 3,
    factor = 0.5,
    min_lr = 1e-5
)

import cv2
import numpy as np

def generate_augmented_data(images, labels, batch_size=128):
    while True:
        batch_indices = np.random.choice(len(images), size=batch_size, replace=False)
        batch_images = []
        batch_labels = []
        for idx in batch_indices:
            image = images[idx]
            label = labels[idx]
            # Apply augmentation using OpenCV
            # Example: rotate image by a random angle between -10 and 10 degrees
            angle = np.random.uniform(-10, 10)
            rotated_image = rotate_image(image, angle)
            # Other augmentations such as zoom, shift, etc. can be applied similarly
            
            batch_images.append(rotated_image)
            batch_labels.append(label)
        
        yield np.array(batch_images), np.array(batch_labels)

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # Apply rotation
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

# Generate augmented training data
trainGen = generate_augmented_data(trainX, trainY)
# Generate validation data (without augmentation)
validGen = generate_augmented_data(testX, testY)
# Generate test data (without augmentation)
testGen = generate_augmented_data(testImages, testLabels)

# Define number of epochs
epochs = 29

history = model.fit(
    trainGen,
    steps_per_epoch=len(trainX) // 32,  # Number of batches per epoch
    validation_data=validGen,
    validation_steps=len(testX) // 32,  # Number of batches in validation set
    epochs=epochs,
    callbacks=[earlyStopping, reduceLR],
  
)


# Save the trained model in the native Keras format
model.save('sign_language_model.keras')














