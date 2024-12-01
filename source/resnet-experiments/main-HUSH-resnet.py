
#https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import numpy as np
tf.get_logger().setLevel('ERROR')
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
import numpy as np
from math import ceil
tf.keras.backend.clear_session()
import keras.backend as K

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.utils.np_utils import to_categorical


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_shards(data, seg_count):
    return np.array_split(data, seg_count)

def load_data():
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    y_train_ohe = to_categorical(train_labels, num_classes = 10)
    y_test_ohe = to_categorical(test_labels, num_classes = 10)

    X_train = train_data.astype('float32')
    X_test = test_data.astype('float32')
    X_train  /= 255
    X_test /= 255

    return X_train, y_train_ohe, X_test, y_test_ohe

def build_model():
    # model = resnet_v2(input_shape=input_shape, depth=depth)
    model= tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax') 
])
    return model

def get_opt_loss():
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    loss = 'categorical_crossentropy'
    
    return optimizer, loss

def train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size, aug):
    # hist = model.fit(train_data, train_labels,
    #         epochs=epochs,
    #         validation_data=(test_data, test_labels),
    #         batch_size=batch_size)
    
    hist = model.fit_generator(
        aug.flow(train_data,train_labels, batch_size=batch_size),
        # validation_data=(test_data, test_labels),
        validation_split=0.2,
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs)
    
    return model, hist

def test_model(model, test_data, test_labels):
    score = model.evaluate(test_data, test_labels, verbose=0)
    test_loss = score[0]
    test_accuracy = score[1]
    return model, test_loss, test_accuracy

def freeze_trunk(model, trunk_ratio):
    for i in range(ceil(len(model.layers)*trunk_ratio)):
        model.layers[i].trainable = False
    return model



def dp_main():

    batch_size = 128  
    epochs = 20
    num_classes = 10

    n = 2
    version = 2
    depth = n * 9 + 2

    aug = ImageDataGenerator(
        rotation_range=20, 
        zoom_range=0.15, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.15,
        horizontal_flip=True, 
        fill_mode="nearest") 
    
    train_data, train_labels, test_data, test_labels = load_data()
    input_shape = train_data.shape[1:]
    HUSH_avg_acc = list()
    baseline_acc = list()

    model = build_model()

    optimizer, loss = get_opt_loss()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model, history = train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size, aug)

    baseline_acc.append(history.history['val_accuracy'][-1]) 

    # HUSH  

    # seg_count = 5
    seg_counts = range(1, 15, 1)
    epochs = 20
    trunk_ratio = 0.25


    for seg_count in seg_counts:
        temp_model = tf.keras.models.clone_model(model)
        temp_model.set_weights(model.get_weights())
        layer_count = ceil(len(temp_model.layers)*trunk_ratio)
        trunk_model = tf.keras.models.Sequential(temp_model.layers[:layer_count])
        constituent_models = list(range(seg_count))

        train_data_shards = get_shards(train_data, seg_count)
        train_label_shards = get_shards(train_labels, seg_count)

        val_acc_sum = 0

        for i in range(seg_count):
            temp_branch = build_model()

            layer_count = ceil(len(temp_branch.layers)*(trunk_ratio))
            print('the second half layer count is : ', len(temp_branch.layers) - layer_count)
            # layer_count = len(temp_branch.layers) - layer_count
            branch_model = tf.keras.models.Sequential(temp_branch.layers[layer_count:])
            
            constituent_models[i] = tf.keras.models.Sequential([
            trunk_model,
            branch_model
            ])

            opt = tf.keras.optimizers.Adam(0.001)
            opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

            constituent_models[i].compile(optimizer=opt, loss=loss, metrics=['accuracy'])
            constituent_models[i], history = train_model(constituent_models[i], 
                                                        train_data_shards[i], 
                                                        train_label_shards[i], 
                                                        epochs, 
                                                        test_data, 
                                                        test_labels, 
                                                        batch_size,
                                                        aug)

            val_acc_sum += history.history['val_accuracy'][-1]
            print(history.history['val_accuracy'][-1]) 

        average_val_acc = val_acc_sum / seg_count
        HUSH_avg_acc.append(average_val_acc)

    print(HUSH_avg_acc)
    print(baseline_acc)

if __name__ == "__main__":
    dp_main()
