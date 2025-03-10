# cnn_model.py
import tensorflow as tf  # Import the full TensorFlow module

# Diagnostic check
print("TensorFlow version:", tf.__version__)
if not hasattr(tf, 'keras'):
    raise ImportError("Keras is not available in this TensorFlow installation.")

from tensorflow.keras import layers, Model  # Import Keras layers and Model


def build_embedding_model(input_shape=(128, 128, 3), embedding_size=128):
    inputs = tf.keras.Input(shape=input_shape)
    

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    embeddings = layers.Dense(embedding_size, activation=None)(x)
    # Use Lambda layer to apply l2_normalize
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
    # - Lambda wraps the TensorFlow operation, making it compatible with Keras

    model = Model(inputs, embeddings)
    return model


# Build and store the model
embedding_model = build_embedding_model()
print("Model built successfully!")