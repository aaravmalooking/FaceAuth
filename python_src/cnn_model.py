
import tensorflow as tf

# Diagnostic check
print("TensorFlow version:", tf.__version__)
if not hasattr(tf, 'keras'):
    raise ImportError("Keras is not available in this TensorFlow installation.")

from tensorflow.keras import layers, Model


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

    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)


    model = Model(inputs, embeddings)
    return model



embedding_model = build_embedding_model()
print("Model built successfully!")