# preparer.py
import cv2  # For image loading and optional face detection
import numpy as np  # For array operations
import os  # For file and folder navigation


# Function to load and preprocess one image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional face detection (since images are pre-aligned)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        img = img[y:y + h, x:x + w]

    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img


# Load the dataset (limit to 100 people for now)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset = {}
data_dir = r"C:\Users\Aarav Maloo\Desktop\FaceAuth\dataset\lfw-deepfunneled"
for person_name in list(os.listdir(data_dir)):  # Limit to 100 people
    person_dir = os.path.join(data_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    dataset[person_name] = []
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = load_and_preprocess_image(img_path)
        if img is not None:
            dataset[person_name].append(img)

# Convert to arrays
X = []
y = []
for person_id, (person_name, images) in enumerate(dataset.items()):
    for img in images:
        X.append(img)
        y.append(person_id)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} images across {len(dataset)} people.")

# Make X and y available to other files
dataset_data = {'X': X, 'y': y, 'dataset': dataset}