
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

def get_shards(data, seg_count):
    return np.array_split(data, seg_count)

def load_data():
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return train_data, train_labels, test_data, test_labels

def build_model():
    return tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 8,
                                strides=2,
                                padding='same',
                                activation='relu',
                                input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Conv2D(32, 4,
                                strides=2,
                                padding='valid',
                                activation='relu'),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

def get_opt_loss():
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    
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
    epochs = 10
    batch_size = 250 

    train_data, train_labels, test_data, test_labels = load_data()

    HUSH_avg_acc = list()
    baseline_acc = list()

    model = build_model()

    optimizer, loss = get_opt_loss()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model, history = train_model(model, train_data, train_labels, epochs, test_data, test_labels, batch_size)

    model, test_loss, test_accuracy = test_model(model, test_data, test_labels)
    # baseline_acc.append(history.history['val_accuracy'][-1]) 
    baseline_acc.append(test_accuracy)

    # HUSH


    # seg_count = 5
    seg_counts = range(1, 4, 1)
    epochs = 10
    trunk_ratio = 0.25


    for seg_count in seg_counts:
        print('starting training for ', seg_count, ' segments:')
        temp_model = tf.keras.models.clone_model(model)
        temp_model.set_weights(model.get_weights())
        layer_count = ceil(len(temp_model.layers)*trunk_ratio)
        trunk_model = tf.keras.models.Sequential(temp_model.layers[:layer_count])
        constituent_models = list(range(seg_count))

        train_data_shards = get_shards(train_data, seg_count)
        train_label_shards = get_shards(train_labels, seg_count)

        val_acc_sum = 0

        for i in range(seg_count):
            print('training constituent model ', i, ' of ', seg_count)
            temp_branch = build_model()

            layer_count = ceil(len(temp_branch.layers)*(trunk_ratio))
            print('the second half layer count is : ', len(temp_branch.layers) - layer_count)
            # layer_count = len(temp_branch.layers) - layer_count
            branch_model = tf.keras.models.Sequential(temp_branch.layers[layer_count:])
            
            constituent_models[i] = tf.keras.models.Sequential([
            trunk_model,
            branch_model
            ])

            opt = tf.keras.optimizers.SGD(0.001)

            constituent_models[i].compile(optimizer=opt, loss=loss, metrics=['accuracy'])
            constituent_models[i], history = train_model(constituent_models[i], 
                                                        train_data_shards[i], 
                                                        train_label_shards[i], 
                                                        epochs, 
                                                        test_data, 
                                                        test_labels, 
                                                        batch_size)

            model, test_loss, test_accuracy = test_model(constituent_models[i], test_data, test_labels)

            # val_acc_sum += history.history['val_accuracy'][-1]
            # print(history.history['val_accuracy'][-1]) 
            val_acc_sum += test_accuracy
            print(test_accuracy)


        average_val_acc = val_acc_sum / seg_count
        HUSH_avg_acc.append(average_val_acc)

    print(HUSH_avg_acc)
    print(baseline_acc)

if __name__ == "__main__":
    dp_main()
