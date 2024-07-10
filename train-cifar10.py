"""
    Script to train RN0X models with CIFAR10 dataset. 
    
    Usage:
        python train-cifar10.py -c config/RN06-Poke10.yml   # RN06
        python train-cifar10.py -c config/RN08-Poke10.yml   # RN08
"""
import os
if os.system('nvidia-smi') == 0:
    import setGPU
import tensorflow as tf
import glob
import sys
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras 
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import resnet_v1_eembc
import yaml
import csv
import json
import datetime
import numpy as np 
import pickle
# import kerop
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
if os.system('nvidia-smi') == 0:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device = "/GPU:0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    device = "/CPU:0"


def rgb_to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

def get_lr_schedule_func(initial_lr, lr_decay):

    def lr_schedule_func(epoch):
        return initial_lr * (lr_decay ** epoch)

    return lr_schedule_func


def main(args):
    ############################################### 
    # parameters
    ############################################### 
    input_shape = [32, 32, 3]
    num_classes = 10 #151
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    loss = config['fit']['compile']['loss']
    model_file_path = os.path.join(f'{save_dir}_cifar10', 'model_best.h5')

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config['model']['final_activation'])

    # optimizer
    optimizer = getattr(tf.keras.optimizers, config['fit']['compile']['optimizer'])
    initial_lr = config['fit']['compile']['initial_lr']
    lr_decay = config['fit']['compile']['lr_decay']
    

    ############################################### 
    # dataset 
    ###############################################  
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip = True,
        validation_split=0.25
        # preprocessing_function=random_crop,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train/256., x_test/256.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    ############################################### 
    # Define model 
    ############################################### 
    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p,
              'skip': skip,
              'avg_pooling': avg_pooling}

    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    # layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # visualize FLOPs results
    # total_flop = 0
    # for name, flop, shape in zip(layer_name, layer_flops, inshape):
    #     print("layer:", name, shape, " MFLOPs:", flop/1e6)
    #     total_flop += flop
    # print("Total FLOPs: {} MFLOPs".format(total_flop/1e6))
    '''
    tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=False,
                              rankdir="TB",
                              expand_nested=False)
    '''

    # compile model with optimizer
    model.compile(
        optimizer=optimizer(learning_rate=initial_lr),
        loss=loss,
        metrics=['accuracy']
    )


    ############################################### 
    # callbacks
    ############################################### 
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

    lr_schedule_func = get_lr_schedule_func(initial_lr, lr_decay)
    log_dir = "logs/cifar10/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True),
                 EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
                 LearningRateScheduler(lr_schedule_func, verbose=verbose),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 ]

    ############################################### 
    # train
    ############################################### 
    if args.evaluate is None:
        with tf.device(device):
            print("############################################### ")
            print('Using CIFAR10 Data')
            print("############################################### ")
            history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=num_epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks,
                        verbose=verbose)    

        # save training history 
        history_dict = history.history
        with open(os.path.join(log_dir,"history.json"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    ############################################### 
    # Evaluate best model 
    ############################################### 
    # restore "best" model
    if args.evaluate is None:
        model.load_weights(model_file_path)
    else:
        model.load_weights(args.evaluate)

    y_pred = model.predict(x_test)
    evaluation = accuracy_score(y_true=np.argmax(y_test, 1), y_pred=np.argmax(y_pred, 1))
    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

    print('Model test accuracy = %.3f' % evaluation)
    print('Model test weighted average AUC = %.3f' % auc)

    model.save(f'{save_dir}_cifar10/model_best_ac{evaluation:0.3f}_au{auc:0.3f}.keras')

    all1 = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1)>0.5)  # TODO: save confusion matrix 
    sns.heatmap(all1, annot=True)
    plt.title("Confusion Matrix - RN0X CIFAR10")
    plt.xlabel("Predicted Class (>0.5)")
    plt.ylabel("True Class")
    plt.show()
    plt.savefig(f'{save_dir}_cifar10/RN0X_CIFAR10_ConfMat.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  training options 
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")
    # eval 
    parser.add_argument('-e', '--evaluate', type=str, default=None, help="Evaluate given model, does not train")

    args = parser.parse_args()

    main(args)
