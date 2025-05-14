#!/usr/bin/env python3
"""
5-transformer.py

Defines the Transformer model for sequence-to-sequence translation
(Portuguese to English).
"""

import tensorflow as tf
import numpy as np


def get_angles(pos, i, dm):
    """
    Computes the angles for positional encoding.
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    return pos * angle_rates


def positional_encoding(max_len, dm):
    """
    Returns a (1, max_len, dm) positional encoding tensor.
    """
    angle_rads = get_angles(
        np.arange(max_len)[:, np.newaxis],
        np.arange(dm)[np.newaxis, :],
        dm
    )
    # apply sin to even indices in the array; cos to odd
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.
    """

    def __init__(self, dm, h):
        super().__init__()
        assert dm % h == 0
        self.h = h
        self.dh = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x):
        """
        Splits the last dimension into (h, dh) and transposes the result.
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.h, self.dh))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        scaled = tf.matmul(attention_weights, V)
        scaled = tf.transpose(scaled, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled, (tf.shape(scaled)[0], -1, self.h * self.dh))
        return self.linear(concat)


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, dm, hidden):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense2 = tf.keras.layers.Dense(dm)

    def call(self, x):
        return self.dense2(self.dense1(x))


class EncoderLayer(tf.keras.layers.Layer):
    """
    Single layer of the encoder.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = PointWiseFeedForward(dm, hidden)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attn = self.mha(x, x, x, mask)
        attn = self.dropout1(attn, training=training)
        out1 = self.norm1(x + attn)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(out1 + ffn_out)


class DecoderLayer(tf.keras.layers.Layer):
    """
    Single layer of the decoder.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.ffn = PointWiseFeedForward(dm, hidden)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.drop3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.drop1(attn1, training=training)
        out1 = self.norm1(attn1 + x)
        attn2 = self.mha2(out1, enc_out, enc_out, padding_mask)
        attn2 = self.drop2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)
        ffn_out = self.ffn(out2)
        ffn_out = self.drop3(ffn_out, training=training)
        return self.norm3(ffn_out + out2)


class Encoder(tf.keras.layers.Layer):
    """
    Stacks N encoder layers.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_len, drop_rate=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.pos_encoding = positional_encoding(max_len, dm)
        self.enc_layers = [EncoderLayer(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Stacks N decoder layers.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_len, drop_rate=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.pos_encoding = positional_encoding(max_len, dm)
        self.dec_layers = [DecoderLayer(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_out, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """
    Full Transformer model.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_len,
                 drop_rate=0.1):
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_len, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_len, drop_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inp, tar, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_out = self.encoder(inp, training, enc_padding_mask)
        dec_out = self.decoder(
            tar, enc_out, training, look_ahead_mask, dec_padding_mask
        )
        return self.final_layer(dec_out)
