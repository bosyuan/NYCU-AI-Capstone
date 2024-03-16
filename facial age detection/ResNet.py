import numpy as np
import cv2
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from matplotlib.pyplot import show
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.transform import rescale

# Function to preprocess labels into 10-year bins
def preprocess_labels(age):
    label = int((age / 10) - 2)  # Group ages into 10-year bins
    return label

# Function to load images from directory
def load_images_from_dir(directory, augmentation=False):
    images = []
    labels = []
    for folder in os.listdir(directory):
        age = int(folder)
        print(f"Loading images from {folder}...")
        label = preprocess_labels(age)
        folder_path = os.path.join(directory, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(os.path.join(folder_path, filename))  # Load image with cv2
                img = cv2.resize(img, (168, 224))  # Resize image to 120 x 160 with cv2
                images.append(img)
                labels.append(label)  # Use preprocessed label

                # Data augmentation
                if augmentation:
                    # Rotate image
                    rotated_img = rotate(img, angle=np.random.uniform(-15, 15), mode='edge')
                    images.append(rotated_img)
                    labels.append(label)
                    
                    # Flip image horizontally
                    flipped_img = cv2.flip(img, 1)
                    images.append(flipped_img)
                    labels.append(label)
                    
                    # Add Gaussian noise
                    noisy_img = random_noise(img, var=0.01**2)
                    noisy_img = (255*noisy_img).astype(np.uint8)
                    images.append(noisy_img)
                    labels.append(label)
                    
                    # Rescale image
                    scaled_img = rescale(img, scale=np.random.uniform(0.8, 1.2), mode='constant')
                    images.append((scaled_img * 255).astype(np.uint8))
                    labels.append(label)
                
    return np.array(images), np.array(labels)

# Load images from your dataset directory
dataset_dir = 'dataset'
X, y = load_images_from_dir(dataset_dir)

# Initialize variables for cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred = []
y_true = []

# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    print("start iteration: " + str(test_index[0]))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Load pre-trained ResNet50 model without top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 168, 3))

    # Add Global Average Pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add Dense layer for classification (adjust the number of units according to your needs)
    predictions = Dense(5, activation='softmax')(x)

    # Combine base model and new layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze pre-trained layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Predict on test set
    y_pred_fold = np.argmax(model.predict(X_test), axis=1)
    
    y_pred.extend(y_pred_fold)
    y_true.extend(y_test)

    # Print accuracy of individual test/train set partition
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    break

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
show()

# Calculate accuracy from confusion matrix
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print("Accuracy:", accuracy)
