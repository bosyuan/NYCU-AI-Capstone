import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib.pyplot import show
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

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

# Perform cross-validation with progress bar
y_pred = []
with tqdm(total=10) as pbar:
    for i, (train_idx, test_idx) in enumerate(cross_val_predict(model, X, y, cv=10, method='predict_proba')):
        model.fit(X[train_idx], y[train_idx])
        y_pred.append(model.predict(X[test_idx]))
        pbar.update(1)

y_pred = np.concatenate(y_pred)

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
