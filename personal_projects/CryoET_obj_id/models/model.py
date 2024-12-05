from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam

set_global_policy('mixed_float16')

print("Mixed precision policy enabled:")

def unet_3d(input_shape):
    """
    3D U-Net model.
    Args:
    - input_shape: Tuple representing the shape of input (depth, height, width, channels).
    Returns:
    - model: Compiled 3D U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(128, (3, 3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = Conv3D(256, (3, 3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(256, (3, 3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder
    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv3D(128, (3, 3, 3), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(128, (3, 3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv3D(64, (3, 3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(64, (3, 3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv3D(32, (3, 3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(32, (3, 3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid', dtype='float32')(conv7)

    optimizer = Adam()

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
