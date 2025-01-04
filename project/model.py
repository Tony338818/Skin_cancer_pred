import numpy as np
import cv2 as cv
import os
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization


dataset_dir = 'c:/Users/Dell/Downloads/archive (1)/melanoma_cancer_dataset/train'
IMG_SIZE = (128, 128) 


def load_images(folder_path, label):
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = cv.imread(file_path)
        if image is not None: 
            image = cv.resize(image, IMG_SIZE)
            images.append(image)
            labels.append(label)
        else:
            print(f"Warning: Unable to load image {file_path}")
    return images, labels


print('Starting image reading')
# Load training malignant images (label = 1)
malignant_train_folder = os.path.join(dataset_dir, 'Malignant')
malignant_train_images, malignant_train_labels = load_images(malignant_train_folder, label=1)

# Load training benign images (label = 0)
benign_train_folder = os.path.join(dataset_dir, 'Benign')
benign_train_images, benign_train_labels = load_images(benign_train_folder, label=0)

# Load test malignant images (label = 1)
malignant_test_folder = os.path.join(dataset_dir, 'Malignant')
malignant_test_images, malignant_test_labels = load_images(malignant_test_folder, label=1)

# Load test benign images (label = 0)
benign_test_folder = os.path.join(dataset_dir, 'Benign')
benign_test_images, benign_test_labels = load_images(benign_test_folder, label=0)

print('Done reading images')
# Combine malignant and benign data
x_train = np.array(malignant_train_images + benign_train_images)
y_train = np.array(malignant_train_labels + benign_train_labels)
x_test = np.array(malignant_test_images + benign_test_images)
y_test = np.array(malignant_test_labels + benign_test_labels)

x_train, x_test = x_train/255.0, x_test/255.0
# y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)



print('building model')
# build the model
model = models.Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    Conv2D(32, (3, 3), activation='relu'),
    Flatten(),
    # Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])


print('compiling the model')
# compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print('training the model')
# train the model
model.fit(x_train, y_train, epochs=20)

# test the model
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'test accuracy is : {test_acc}')

# saving the model
model.save('Skin_Cancer_model.h5')  



