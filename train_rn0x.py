"""
    Script to train RN0X models with Pokemon dataset. Use pretrained CIFAR10 RN0X models 
    for best performance. 

    Options:
        --config (str)   : path to yaml file 
        --pretrain (str) : path to pretrained model (not weights only)
        --freeze (bool)  : freeze all layers but the last 
        --re-init (bool) : re-initializes the last layer (only when 'freeze' option is given)
    
    Usage:
            python train_rn0x.py -c config/RN06-Poke10.yml -p resnet_v1_eembc_RN06_cifar10/model_best.h5 --freeze --data 2024 
            python train_rn0x.py -c config/RN06-Poke10.yml -p resnet_v1_eembc_RN06_cifar10/model_best.h5 --freeze --re-init

            python train_rn0x.py -c config/RN08-Poke10.yml -p resnet_v1_eembc_RN08_cifar10/model_best.h5 --freeze
            python train_rn0x.py -c config/RN08-Poke10.yml -p resnet_v1_eembc_RN08_cifar10/model_best.h5 --freeze --re-init
"""
import os
if os.system('nvidia-smi') == 0:
    import setGPU
import tensorflow as tf
import logging 
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import keras.backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import resnet_v1_eembc
import yaml
import datetime
import pickle
# import kerop
from sklearn.metrics import confusion_matrix
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

def load_and_preprocess_image(image_path, target_size):
    """
    Load and preprocess an image.

    Args:
    image_path (str): Path to the image file.
    target_size (tuple): Target size for the image (width, height).

    Returns:
    numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Load the image
    img = image.load_img(image_path, target_size=target_size)

    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the shape required by the model (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (if required, depending on your model's training)
    img_array = img_array / 255.0

    return img_array


def predict_image(model, image_path):
    test_image = load_and_preprocess_image(
        image_path,
        (32, 32)
    )
    return np.argmax(model.predict(test_image), 1)


def setup_train_logger(save_dir):
    # Create a logger
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)  # Set the desired log level

    # Create handlers for both stdout and the log file
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the desired log level for the console

    file_handler = logging.FileHandler(os.path.join(save_dir, 'train_rn0x.log'))
    file_handler.setLevel(logging.DEBUG)  # Set the desired log level for the file

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger 


def custom_categorical_accuracy(y_true, y_pred):
    """
    Custom categorical accuracy metric function.

    Args:
    y_true: Tensor of true labels.
    y_pred: Tensor of predicted labels.

    Returns:
    accuracy: Tensor representing the accuracy of the predictions.
    """
    # Convert probabilities to predicted class
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)

    # Check if predictions are equal to the true labels
    correct_predictions = K.equal(y_true_classes, y_pred_classes)

    # Cast boolean values to float and calculate mean accuracy
    accuracy = K.mean(K.cast(correct_predictions, K.floatx()))

    return accuracy


def main(args):
    ############################################### 
    # yml parameters
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

    save_dir = f'checkpoints/{save_dir}_{args.dataset}'
    if args.freeze and args.re_init:
        save_dir = f'{save_dir}_fr'
    elif args.freeze:
        save_dir = f'{save_dir}_fo'
    elif args.re_init:
        save_dir = f'{save_dir}_ro'

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    else:
        print('====================================================')
        print(f'WARNING: Checkpoint directory {save_dir} already exists.')
        print('====================================================')
    
    logger = setup_train_logger(save_dir)
    logger.info('Checkpoint directory set to', save_dir)
    logger.info('Train RN0X Arguements:', args)
    logger.info('Config Parameters:', config)
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
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.25,
        preprocessing_function=scale_image,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )
    
    train_generator = datagen.flow_from_directory(f'./data/{args.dataset}/',
        target_size=(32,32),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        interpolation='bilinear',
        subset='training',
        # shuffle=True,
        #keep_aspect_ratio=True,                                             
    )
    
    validation_generator = datagen.flow_from_directory(f'./data/{args.dataset}/',
        target_size=(32,32),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        interpolation='bilinear',
        subset='validation',
        # shuffle=True
        #keep_aspect_ratio=True
    )

    print('=================================')
    print('Batches in train : ', len(train_generator))
    print('Batches in test  : ', len(validation_generator))
    print('=================================')


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
    model.load_weights(os.path.join('checkpoints', args.pretrain))

    logger.info('##################################')
    logger.info('#          MODEL SUMMARY         #')
    logger.info('##################################')
    logger.info(model.summary())
    logger.info('##################################')

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
    tensorboard_log_dir = os.path.join("logs/rn0x_fit_10/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    callbacks = [
        ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True, save_weights_only=False),
        EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
        LearningRateScheduler(lr_schedule_func, verbose=verbose),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1),
    ]

    ############################################### 
    # train
    ############################################### 
    if args.freeze:
        # freeze all layers, unfreeze last layer 
        model.trainable = False
        logger.info(f'Freezing layer {model.layers[-2]}')
        model.layers[-2].trainable = True

        if args.re_init:  # https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
            logger.info(f'Reinitializing layer {model.layers[-2]}')
            new_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in model.layers[-2].get_weights()]
            model.layers[-2].set_weights(new_weights) # re-initialize last layer weights 

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
    logger.info('Training completed.')

    with open(os.path.join(tensorboard_log_dir,"history.json"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ############################################### 
    # Evaluate best model 
    ############################################### 
    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)
    # restore "best" model
    logger.info(f'Loading weights: {model_file_path}')
    model.load_weights(model_file_path)

    # compile model with optimizer
    model.compile(
        optimizer=optimizer(learning_rate=initial_lr),
        loss=loss,
        metrics=['accuracy']
    )

    # get predictions
    y_pred_val = model.predict(train_generator)
    logger.info("Keras Predict Accuracy:  {}".format(accuracy_score(
        np.argmax(y_pred_val, axis=1), train_generator.classes)
        )
    )

    # evaluate with test dataset and share same prediction results
    val_evaluation = model.evaluate(train_generator)     ######
    train_evaluation = model.evaluate(train_generator)

    auc = roc_auc_score(train_generator.classes, y_pred_val, average='weighted', multi_class='ovr')    ######

    logger.info('Model train accuracy = %.3f' % val_evaluation[1])
    logger.info('Model test accuracy = %.3f' % train_evaluation[1])
    logger.info('Model test weighted average AUC = %.3f' % auc)

    model.save(f'{save_dir}/model_best_ac{val_evaluation[1]:0.3f}_au{auc:0.3f}.keras')

    # all1 = confusion_matrix(validation_generator.classes, np.argmax(y_pred, 1)>0.5)
    all1 = confusion_matrix(train_generator.classes, np.argmax(y_pred_val, 1))     ######
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
    parser.add_argument('-d', '--dataset', choices=['cam', '2024', '05302024', '05302024_aug'], default='cam', help="choose a dataset")
    parser.add_argument('-f', '--freeze', action="store_true", default=False, help="freeze all but last layer")
    parser.add_argument('-i', '--re-init', action="store_true", default=False, help="re-initialize last layer weights")
    # evaluate 
    parser.add_argument('-e', '--evaluate', type=str, default=None, help="evaluate given model, does not train")  # TODO

    args = parser.parse_args()

    main(args)
