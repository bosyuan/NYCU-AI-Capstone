import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Function to preprocess labels into 5-year bins
def preprocess_labels(age):
    label = int((age / 10) - 2)  # Group ages into 5-year bins
    return label

# Function to load images from directory
def load_images_from_dir(directory):
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
                img = cv2.resize(img, (120, 160))  # Resize image to 120 x 160 with cv2
                img_array = np.array(img).flatten()  # Flatten image to 1D array
                images.append(img_array)
                labels.append(label)  # Use preprocessed label
    return np.array(images), np.array(labels)

# Load images from your dataset directory
dataset_dir = 'dataset'
X, y = load_images_from_dir(dataset_dir)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Model evaluation
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
