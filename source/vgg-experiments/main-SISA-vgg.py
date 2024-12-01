
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
import numpy as np
tf.get_logger().setLevel('ERROR')
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
import numpy as np
from math import ceil
tf.keras.backend.clear_session()

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


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
    vgg16_model = VGG16(weights='imagenet', #One of None (random initialization), 'imagenet' (pre-training on ImageNet)
                    include_top=False, 
                    classes=10,
                    input_shape=(32,32,3)# input: 32x32 images with 3 channels -> (32, 32, 3) tensors.
                   )
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    from tensorflow.keras.layers import Dense, Flatten, Dropout
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='hidden1'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu', name='hidden2'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax', name='predictions'))

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation='softmax')
    #     ])
    return model


def train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size, aug):
    # hist = model.fit_generator(
    #     aug.flow(train_data,train_labels, batch_size=batch_size),
    #     validation_data=(test_data, test_labels),
    #     steps_per_epoch=len(train_data) // batch_size,
    #     epochs=epochs)
    hist = model.fit(train_data, train_labels,
            epochs=epochs,
            # validation_data=(test_data, test_labels),
            validation_split=0.2,
            batch_size=batch_size)
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
    epochs = 20
    batch_size = 125

    l2_norm_clip = 1.5
    noise_multiplier = np.arange(0.1, 2, 0.1)
    # noise_multiplier = [1.3]
    # num_microbatches = 128
    # learning_rate = 0.25

    model_built = False

    aug = ImageDataGenerator(
        rotation_range=20, 
        zoom_range=0.15, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.15,
        horizontal_flip=True, 
        fill_mode="nearest") 
    
    # if batch_size % num_microbatches != 0:
    #     raise ValueError('Batch size should be an integer multiple of the number of microbatches')  

    train_data, train_labels, test_data, test_labels = load_data()

    SISA_dp_avg_acc = list()
    # seg_count = 5
    seg_counts = range(2, 17, 1)
    
    for seg_count in seg_counts:
        print('starting training for ', seg_count, ' segments:')
        constituent_models = list(range(seg_count))
        train_data_shards = get_shards(train_data, seg_count)
        train_label_shards = get_shards(train_labels, seg_count)
    
   
        val_acc_sum = 0

        # SISA 
        for i in range(seg_count):
            print('training constituent model ', i+1, ' of ', seg_count)
            constituent_models[i] = build_model()
            # opt = tf.keras.optimizers.SGD(learning_rate = 0.01,
            #                                 momentum=0.0, 
            #                                 nesterov=False, 
            #                                 name='SGD')
            # opt = tf.keras.optimizers.Adam(0.001)
            opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

            loss = 'categorical_crossentropy'
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
        
        average_val_acc = val_acc_sum/seg_count
        SISA_dp_avg_acc.append(average_val_acc)
    
    print(SISA_dp_avg_acc)
if __name__ == "__main__":
    dp_main()
