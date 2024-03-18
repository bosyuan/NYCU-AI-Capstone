import numpy as np
import cv2
import os
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import show

# Function to preprocess labels into 10-year bins
def preprocess_labels(age):
    label = int((age / 10) - 2)  # Group ages into 10-year bins
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

# Initialize logistic regression model
model = LogisticRegression(max_iter=2000)

# Perform cross-validation and get predicted labels
y_pred = cross_val_predict(model, X, y, cv=2)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

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