# trainer.py
from preparer import dataset_data
from cnn_model import embedding_model
from triplet_loss import triplet_loss
import numpy as np


X = dataset_data['X']
y = dataset_data['y']
dataset = dataset_data['dataset']



def generate_triplets(X, y, num_triplets=1000):
    triplets = []
    num_classes = len(np.unique(y))
    for _ in range(num_triplets):
        pos_class = np.random.randint(0, num_classes)
        pos_indices = np.where(y == pos_class)[0]
        if len(pos_indices) < 2:
            continue
        anchor_idx, pos_idx = np.random.choice(pos_indices, 2, replace=False)
        neg_class = np.random.randint(0, num_classes)
        while neg_class == pos_class:
            neg_class = np.random.randint(0, num_classes)
        neg_indices = np.where(y == neg_class)[0]
        if len(neg_indices) == 0:
            continue
        neg_idx = np.random.choice(neg_indices)
        triplets.append([X[anchor_idx], X[pos_idx], X[neg_idx]])
    return np.array(triplets)



triplets = generate_triplets(X, y, num_triplets=1000)
print(f"Generated {len(triplets)} triplets.")
if len(triplets) == 0:
    print("No valid triplets generated—try increasing people or images.")
else:

    batch_size = 30
    num_triplets = len(triplets)
    num_triplets = (num_triplets // batch_size) * batch_size
    triplets = triplets[:num_triplets]
    print(f"Adjusted to {num_triplets} triplets for training.")

    triplet_inputs = triplets.reshape(-1, 128, 128, 3)
    y_dummy = np.zeros((len(triplet_inputs), 128))


    embedding_model.compile(optimizer='adam', loss=triplet_loss)
    embedding_model.fit(triplet_inputs, y_dummy, batch_size=batch_size, epochs=5)
    print("Training complete!")