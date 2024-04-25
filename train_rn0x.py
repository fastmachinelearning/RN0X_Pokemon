"""
    Script to train RN0X models with Pokemon dataset. Use pretrained CIFAR10 RN0X models 
    for best performance. 

    Options:
        --config (str)   : path to yaml file 
        --pretrain (str) : path to pretrained model (not weights only)
        --freeze (bool)  : freeze all layers but the last 
        --re-init (bool) : re-initializes the last layer (only when --freeze option is given)
    
    Usage:
            python train_rn0x.py -c config/RN06-Poke10.yml -p resnet_v1_eembc_RN06_cifar10/model_best.h5 --freeze
            python train_rn0x.py -c config/RN06-Poke10.yml -p resnet_v1_eembc_RN06_cifar10/model_best.h5 --freeze --re-init

            python train_rn0x.py -c config/RN08-Poke10.yml -p resnet_v1_eembc_RN08_cifar10/model_best.h5 --freeze
            python train_rn0x.py -c config/RN08-Poke10.yml -p resnet_v1_eembc_RN08_cifar10/model_best.h5 --freeze --re-init
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
import keras.backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import resnet_v1_eembc
import yaml
import csv
import json
import datetime
import pickle
import kerop
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def scale_image(x):
   return x/256.

def rgb_to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

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
    if args.freeze and args.re_init:
        save_dir = f'{save_dir}_fr'
    elif args.freeze:
        save_dir = f'{save_dir}_fo2'
    elif args.re_init:
        save_dir = f'{save_dir}_ro'
    
    model_file_path = os.path.join(save_dir, 'model_best.h5')

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
        vertical_flip=True,
        validation_split=0.25,
        preprocessing_function=scale_image,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )
    
    train_generator = datagen.flow_from_directory('./data/PokeCard_2024/',
        target_size=(32,32),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        interpolation='bilinear',
        subset='training',
        shuffle=True,
        #keep_aspect_ratio=True,                                             
    )
    
    validation_generator = datagen.flow_from_directory('./data/PokeCard_2024/',
        target_size=(32,32),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        interpolation='bilinear',
        subset='validation',
        shuffle=True
        #keep_aspect_ratio=True
    )

    ############################################### 
    # define model 
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
    # load pretrained weights 
    model.load_weights(args.pretrain)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # visualize FLOPs results
    total_flop = 0
    for name, flop, shape in zip(layer_name, layer_flops, inshape):
        print("layer:", name, shape, " MFLOPs:", flop/1e6)
        total_flop += flop
    print("Total FLOPs: {} MFLOPs".format(total_flop/1e6))
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
    log_dir = "logs/rn0x_fit_10/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True, save_weights_only=False),
                 EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
                 LearningRateScheduler(lr_schedule_func, verbose=verbose),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 ]

    ############################################### 
    # train
    ############################################### 
    if args.freeze:
        # freeze all layers, unfreeze last layer 
        model.trainable = False
        model.layers[-2].trainable = True
        model.layers[1] = True
        
        if args.re_init:  # https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
            new_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in model.layers[-2].get_weights()]
            # re-initialize last layer weights 
            model.layers[-2].set_weights(new_weights)

    with tf.device(device):
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            validation_data = validation_generator, 
            validation_steps = validation_generator.samples // batch_size,
            epochs = num_epochs,
            callbacks=callbacks,
            verbose=verbose
        )
    
    with open(os.path.join(log_dir,"history.json"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ############################################### 
    # Evaluate best model 
    ############################################### 
    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)
    # restore "best" model
    model.load_weights(model_file_path)

    # compile model with optimizer
    model.compile(
        optimizer=optimizer(learning_rate=initial_lr),
        loss=loss,
        metrics=['accuracy']
    )

    # get predictions
    y_pred = model.predict(validation_generator)

    # evaluate with test dataset and share same prediction results
    evaluation = model.evaluate(validation_generator)

    auc = roc_auc_score(validation_generator.classes, y_pred, average='weighted', multi_class='ovr')

    print('Model test accuracy = %.3f' % evaluation[1])
    print('Model test weighted average AUC = %.3f' % auc)

    model.save(f'{save_dir}/model_best_ac{evaluation[1]:0.3f}_au{auc:0.3f}.keras')

    all1 = confusion_matrix(validation_generator.classes, np.argmax(y_pred, 1)>0.5)
    sns.heatmap(all1, annot=True)
    plt.title("Confusion Matrix - RN0X All Gen 1 Pokemon")
    plt.xlabel("Predicted Class (>0.5)")
    plt.ylabel("True Class")
    plt.show()
    plt.savefig(f'{save_dir}/RN0X_Gen1_ConfMat.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  training options 
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config (RN0X type)")
    parser.add_argument('-p', '--pretrain', type=str, default=None, help="specify the pretrained model for transfer learning")
    parser.add_argument('-f', '--freeze', action="store_true", default=False, help="Freeze all but last layer")
    parser.add_argument('-i', '--re-init', action="store_true", default=False, help="Re-initialize last layer weights")
    # evaluate 
    parser.add_argument('-e', '--evaluate', type=str, default=None, help="Evaluate given model, does not train")  # TODO

    args = parser.parse_args()

    main(args)
