
from cnn_model import embedding_model
from preparer import load_and_preprocess_image
import numpy as np


def compute_embedding(image, model):
    img = np.expand_dims(image, axis=0)
    return model.predict(img)[0]



from preparer import dataset_data

dataset = dataset_data['dataset']

known_embeddings = {}
for person_name, images in dataset.items():
    if len(images) > 0:
        embeddings = [compute_embedding(img, embedding_model) for img in images]
        known_embeddings[person_name] = np.mean(embeddings, axis=0)


test_image_path = r"C:\Users\Aarav Maloo\Desktop\FaceAuth\dataset\lfw-deepfunneled\Aaron_Eckhart\test.jpg"  # Replace with your test image
test_face = load_and_preprocess_image(test_image_path)
if test_face is None:
    print("No face detected in test image")
else:
    test_embedding = compute_embedding(test_face, embedding_model)


    def euclidean_distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    min_dist = float('inf')
    best_match = None
    for person_name, embedding in known_embeddings.items():
        dist = euclidean_distance(test_embedding, embedding)
        if dist < min_dist:
            min_dist = dist
            best_match = person_name

    threshold = 0.6
    if min_dist < threshold:
        print(f"Recognized as {best_match} with distance {min_dist}")
    else:
        print("Unknown face")