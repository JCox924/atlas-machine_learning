#!/usr/bin/env python3
"""
5-train.py

Defines `train_transformer` to train the Transformer model
on Portuguese→English translation.
"""

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup.
    """
    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Computes sparse categorical crossentropy loss,
    ignoring padding (token ID == 0).
    """
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss_ = scce(real, pred) * mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a Transformer for pt→en translation.

    Args:
        N (int): number of encoder/decoder blocks.
        dm (int): model dimensionality.
        h (int): number of heads.
        hidden (int): hidden layer size in feed-forward networks.
        max_len (int): maximum sequence length.
        batch_size (int): training batch size.
        epochs (int): number of epochs.

    Returns:
        transformer (Transformer): trained model.
    """
    # Prepare data
    data = Dataset(batch_size, max_len)

    # Vocabulary sizes (+2 for start/end tokens)
    inp_vocab = data.tokenizer_pt.vocab_size + 2
    tar_vocab = data.tokenizer_en.vocab_size + 2

    # Instantiate model
    transformer = Transformer(
        N, dm, h, hidden,
        inp_vocab, tar_vocab,
        max_len
    )

    # Optimizer and schedule
    lr_schedule = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        lr_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        batch_num = 0

        for (inp, tar) in data.data_train:
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, comb_mask, dec_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                preds = transformer(
                    inp,
                    tar_inp,
                    True,
                    enc_mask,
                    comb_mask,
                    dec_mask
                )
                loss = loss_function(tar_real, preds)

            grads = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, transformer.trainable_variables)
            )

            train_loss(loss)
            train_accuracy(tar_real, preds)

            if batch_num % 50 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_num}: "
                    f"Loss {train_loss.result():.6f}, "
                    f"Accuracy {train_accuracy.result():.6f}"
                )
            batch_num += 1

        # End of epoch
        print(
            f"Epoch {epoch}: "
            f"Loss {train_loss.result():.6f}, "
            f"Accuracy {train_accuracy.result():.6f}"
        )

    return transformer
