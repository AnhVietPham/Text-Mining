import numpy as np
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


if __name__ == "__main__":
    mask = tf.constant([[0, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0]], dtype=tf.float64)

    scaled_attention_logit = tf.constant([[1, 3, 10],
                                          [1, 2, 5],
                                          [1, 1, 5]], dtype=tf.float64)

    scaled_attention_logit += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logit, axis=-1)
    print('Mask * -1e9: ', mask * -1e9)
    print('Scaled attention logit: ', scaled_attention_logit)
    print('Attention Weights: ', attention_weights)

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)

    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    out, attn = scaled_dot_product_attention(
        temp_q, temp_k, temp_v, None)

    print('Attention weights:')
    print(attn)
    print('Output is:')
    print(out)
