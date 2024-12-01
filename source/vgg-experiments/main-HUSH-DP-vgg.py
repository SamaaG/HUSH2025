
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
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow_privacy.privacy.optimizers import dp_optimizer
# from tensorflow_privacy.privacy.analysis import privacy_ledger


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

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='hidden1'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu', name='hidden2'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax', name='predictions'))
    return model


def get_opt_loss(l2_norm_clip, noise_multiplier, num_microbatches, learning_rate):
    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)

    # optimizer = tf.keras.optimizers.Adam(0.001)
    
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    loss = 'categorical_crossentropy'
    return optimizer, loss

def train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size):
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
    # noise_multiplier = np.arange(0.1, 2, 0.1)
    noise_multiplier = 0.7
    num_microbatches = 1
    learning_rate = 0.0001


    if batch_size % num_microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')  
    

    train_data, train_labels, test_data, test_labels = load_data()

    HUSH_dp_avg_acc = list()
    baseline_dp_acc = list()

    model = build_model()

    (eps, opt_RDP) = compute_dp_sgd_privacy(n=train_data.shape[0],
                                                batch_size=batch_size,
                                                noise_multiplier=noise_multiplier,
                                                epochs=epochs,
                                                delta=1e-5)
    print(eps)
    optimizer, loss = get_opt_loss(l2_norm_clip, noise_multiplier, num_microbatches, learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model, history = train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size)

    baseline_dp_acc.append(history.history['val_accuracy'][-1]) 


    # HUSH + DP 
    # seg_count = 5
    seg_counts = range(2, 17, 1)
    epochs = 20
    trunk_ratio = 0.25

    model = freeze_trunk(model, trunk_ratio)

    for seg_count in seg_counts:
        print('starting training for ', seg_count, ' segments:')
        constituent_models = list(range(seg_count))

        train_data_shards = get_shards(train_data, seg_count)
        train_label_shards = get_shards(train_labels, seg_count)

        val_acc_sum = 0

        for i in range(seg_count):
            print('training constituent model ', i+1, ' of ', seg_count)
            constituent_models[i] = tf.keras.models.clone_model(model)
            constituent_models[i].set_weights(model.get_weights())
            opt = tf.keras.optimizers.SGD(learning_rate=0.0004, momentum=0.5)

            constituent_models[i].compile(optimizer=opt, loss=loss, metrics=['accuracy'])
            constituent_models[i], history = train_model(constituent_models[i], 
                                                        train_data_shards[i], 
                                                        train_label_shards[i], 
                                                        epochs, 
                                                        test_data, 
                                                        test_labels, 
                                                        batch_size)

            val_acc_sum += history.history['val_accuracy'][-1]
            print(history.history['val_accuracy'][-1]) 

        average_val_acc = val_acc_sum / seg_count
        HUSH_dp_avg_acc.append(average_val_acc)

    print(HUSH_dp_avg_acc)
    print(baseline_dp_acc)

if __name__ == "__main__":
    dp_main()
