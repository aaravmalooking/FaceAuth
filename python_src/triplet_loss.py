# triplet_loss.py
import tensorflow as tf

def triplet_loss(y_true, y_pred, margin=0.2):
    batch_size = tf.shape(y_pred)[0] // 3
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)