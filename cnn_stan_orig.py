
# coding: utf-8

# In[116]:

'''
Works on FER2013 Dataset

'''

from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential


from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils
from keras.callbacks import History


import matplotlib.pyplot as plt

import matplotlib.cm as cm

import os
import numpy as np
import pdb
import argparse
import logging
import h5py




from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import initializers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras import regularizers
from keras import initializers


# In[117]:

IMG_DIM = 48
DATA_DIR = 'data'
DEFAULT_LR = 1e-4
DEFAULT_REG = 0
DEFAULT_NB_EPOCH = 10
DEFAULT_LAYER_SIZE_1 = 32
DEFAULT_LAYER_SIZE_2 = 64
DEFAULT_DROPOUT1 = 0.1
DEFAULT_DROPOUT2 = 0.25
DEFAULT_OUT_DIR = '../outputs/'
DEFAULT_DEPTH1 = 1
DEFAULT_DEPTH2 = 2
DEFAULT_FRAC_POOLING = False
DEFAULT_SAVE_WEIGHTS = False
DEFAULT_USE_BATCHNORM = False

logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)
num_train = 28704
num_val = 3584
train_data_file_default = 'train_all.csv'
val_data_file_default = 'valid_all.csv'

EMOTIONS = [
  'Angry',
  'Disgust',
  'Fear',
  'Happy',
  'Sad',
  'Surprise',
  'Neutral',
]


# In[ ]:




# In[ ]:




# In[118]:

class CNN:
    """
    Convolutional Neural Network model.
    """

    def __init__(self, params={}, verbose=True):
        """
        Initialize the CNN model with a set of parameters.
        Args:
        params: a dictionary containing values of the models' parameters.
        """
        self.verbose = verbose
        self.params = params

        # An empty (uncompiled and untrained) model may be used for visualizations.
        self.empty_model = None

        logging.info('Initialized with params: {}'.format(params))

    def load_data(self, train_data_file, val_data_file, num_train=None, num_val=None):
        """
        Load training and validation data from files.
        Args:
          train_data_file: path to the file containing training examples.
          val_data_file: path to the file containing validation examples.
        """
        logging.info('Reading {} training examples from {}...'.format(num_train, train_data_file))
        self.X_train, self.y_train = self._load_data_from_file(train_data_file, num_train)
        logging.info('Reading {} validation examples from {}...'.format(num_val, val_data_file))
        self.X_val, self.y_val = self._load_data_from_file(val_data_file, num_val)

    def _load_data_from_file(self, filename, num_examples=None):
        if num_examples is None:
            num_examples = sum(1 for line in open(filename))

        X_data = np.zeros((num_examples, 1, 48, 48))
        y_data = np.zeros((num_examples, 1))
        with open(filename, 'r') as f:
            print(filename)
            for i, line in enumerate(f):
                label, pixels = line.split('\t')
                # pixels.split(' ')
                # Reformat image from array of pixels to square matrix.
                #pixels = np.fromstring([int(num) for num in pixels.split(' ')]).reshape((IMG_DIM, IMG_DIM))
                pixels = np.array([float(num) for num in pixels.split(' ')]).reshape((IMG_DIM, IMG_DIM))
                X_data[i][0] = pixels
                y_data[i][0] = int(label)

                if num_examples is not None and i == num_examples - 1:
                    return X_data, y_data

        return X_data, y_data

    def _add_batchnorm_layer(self, model):
        '''
        Add a batch normalization layer to the model if the params specify use batchnorm.
        '''
        if self.params.get('use_batchnorm', DEFAULT_USE_BATCHNORM):
            model.add(BatchNormalization())

    def _get_file_prefix(self):
        file_prefix = ''
        file_prefix += self.params['output_dir']
        param_names = ['lr', 'depth2', 'fractional_pooling', 'use_batchnorm', 'dropout1', 'dropout2']
        for idx, param_name in enumerate(param_names):
            file_prefix += param_name + '=' + str(self.params[param_name])
            if idx < len(param_names) - 1:
                file_prefix += '_'

        return file_prefix

    def _load_weights(self, model):
        weights_path = self.params['weights_path']
        logging.info('Loading weights from file: {}'.format(weights_path))
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # We don't look at the last (fully connected) layers in the savefile.
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)

        f.close()
        logging.info('Weights loaded.')

    def _evaluate(self, model, X_val, Y_val):
        logging.info('Evaluating model...')
        loss, accuracy = model.evaluate(X_val, Y_val, batch_size=128, show_accuracy=True, verbose=1)
        print('loss: ', loss)
        print('accuracy: ', accuracy)

    def _display_wrong_predictions(self, model, X_val, y_val):
        print('Making predictions...')
        # predictions = model.predict_on_batch(X_val[0:10])

        num_to_compare = 50
        num_to_display = 12

        # Get a small number of predicted and actual classes.
        predicted = model.predict_classes(X_val[0:num_to_compare], batch_size=num_to_compare, verbose=1)
        actual = y_val[0:num_to_compare]

        # Store the first `num_to_display` misclassification images and classes.
        misclassified_pixels = [] # Array to store image data of misclassified images.
        misclassifications = [] # Array to store tuples of (predicted, actual) class.

        for i in range(num_to_compare):
            if len(misclassified_pixels) == num_to_display:
                break
            if predicted[i] != actual[i]:
                # A misclassification.
                pixels = X_val[i].reshape(IMG_DIM, IMG_DIM)
                misclassified_pixels.append(pixels)
                misclassifications.append((int(predicted[i]), int(actual[i])))

        rows = 4
        cols = 3

        for idx, pixels in enumerate(misclassified_pixels):
            predicted_emotion = EMOTIONS[misclassifications[idx][0]]
            actual_emotion = EMOTIONS[misclassifications[idx][1]]
            print('Predicted:', predicted_emotion, '; actual:', actual_emotion)
            plt.subplot(rows, cols, idx + 1)
            plt.gca().axis('off')
            plt.imshow(pixels, cmap = cm.Greys_r)
            plt.title('Predicted: ' + predicted_emotion + '\nActual: ' + actual_emotion)

        plt.tight_layout(h_pad=0.01)
        plt.show()

        pdb.set_trace()

    def train(self):
        """
        Train the CNN model.
        """

        batch_size = 128
        nb_classes = 7

        nb_epoch = self.params.get('nb_epoch', DEFAULT_NB_EPOCH)
        lr = self.params.get('lr', DEFAULT_LR)
        reg = self.params.get('reg', DEFAULT_REG)
        nb_filters_1 = self.params.get('nb_filters_1', DEFAULT_LAYER_SIZE_1)
        nb_filters_2 = self.params.get('nb_filters_2', DEFAULT_LAYER_SIZE_2)
        dropout1 = self.params.get('dropout1', DEFAULT_DROPOUT1)
        dropout2 = self.params.get('dropout2', DEFAULT_DROPOUT2)
        depth1 = self.params.get('depth1', DEFAULT_DEPTH1)
        depth2 = self.params.get('depth2', DEFAULT_DEPTH2)
        fractional_pooling = self.params.get('fractional_pooling', DEFAULT_FRAC_POOLING)
        
        # Printing to verify parameters
        
        print(nb_epoch)
        print(lr)
        print(reg)
        print(nb_filters_1)
        print(nb_filters_2)
        print(dropout1)
        print(dropout2)
        print(depth1)
        print(depth2)
        print(fractional_pooling)
        
        if fractional_pooling:
            print("Using fractional max pooling... \n")
        else:
            print("Using standard max pooling... \n")

        save_weights = self.params.get('save_weights', DEFAULT_SAVE_WEIGHTS)

        X_train, y_train = self.X_train, self.y_train
        X_val, y_val = self.X_val, self.y_val

        # Input image dimensions.
        img_rows, img_cols = IMG_DIM, IMG_DIM

        img_channels = 1

        # Convert class vectors to binary class matrices.
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_val = np_utils.to_categorical(y_val, nb_classes)

        model = Sequential()

        weight_init = 'he_normal'

        # Keep track of which convolutional layer we are at.
        conv_counter = 1

        model.add(Conv2D(nb_filters_1, (3, 3),  
                         padding='same',
                         name='conv_%d' % (conv_counter),
                         input_shape=(img_channels,img_rows, img_cols), 
                         data_format="channels_first",
                         kernel_initializer='he_normal'))
        
        self._add_batchnorm_layer(model)
        conv_counter += 1
        model.add(Activation('relu'))

        for i in range(depth1):
            
            model.add(Conv2D(nb_filters_1, (3, 3), 
                             kernel_initializer='he_normal', 
                             padding='same', 
                             kernel_regularizer=regularizers.l2(reg),
                             activity_regularizer=regularizers.l2(reg),
                             name='conv_%d' % (conv_counter),
                             data_format="channels_first"))
            
            self._add_batchnorm_layer(model)
            conv_counter += 1
            model.add(Activation('relu'))
            
            model.add(Conv2D(nb_filters_1, (3, 3), 
                             kernel_initializer='he_normal', 
                             padding='same', 
                             kernel_regularizer=regularizers.l2(reg),
                             activity_regularizer=regularizers.l2(reg),
                             name='conv_%d' % (conv_counter),
                             data_format="channels_first"))
            
            self._add_batchnorm_layer(model)
            conv_counter += 1
            model.add(Activation('relu'))
            
            if fractional_pooling:
                model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
            else:
                model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
            model.add(Dropout(dropout1))

        for i in range(depth2):
            model.add(Conv2D(nb_filters_2, (3, 3),
                             kernel_initializer='he_normal', 
                             padding='same', 
                             kernel_regularizer=regularizers.l2(reg),
                             activity_regularizer=regularizers.l2(reg),
                             data_format="channels_first",
                             name='conv_%d' % (conv_counter)))
            
            self._add_batchnorm_layer(model)
            conv_counter += 1
            model.add(Activation('relu'))
            
            model.add(Conv2D(nb_filters_2, (3, 3), 
                             kernel_initializer='he_normal', 
                             padding='same', 
                             kernel_regularizer=regularizers.l2(reg),
                             activity_regularizer=regularizers.l2(reg),
                             data_format="channels_first",
                             name='conv_%d' % (conv_counter)))
            
            self._add_batchnorm_layer(model)
            conv_counter += 1
            model.add(Activation('relu'))
            
            if fractional_pooling:
                model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
            else:
                model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
            model.add(Dropout(dropout1))

        model.add(Flatten(input_shape=(img_rows, img_cols)))

        # Add 3 fully connected layers.
        dense_sizes = None
        
        if depth1 == 0 and depth2 == 0:
            print('Running baseline with just 1 dense layer.')
            dense_sizes = [512]
        else:
            dense_sizes = [512, 256, 128]
        
        for idx, dense_size in enumerate(dense_sizes):
            model.add(Dense(dense_size))
            self._add_batchnorm_layer(model)
            model.add(Activation('relu'))

            # Use dropout2 only for the final dense layer.
            if idx == len(dense_sizes) - 1:
                model.add(Dropout(dropout2))
            else:
                model.add(Dropout(dropout1))

        model.add(Dense(nb_classes, 
                        kernel_regularizer=regularizers.l2(reg),
                        activity_regularizer=regularizers.l2(reg)))
        
        model.add(Activation('softmax'))

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_train /= 255
        X_val /= 255

        if False:
            # Load weights and evaluate model immediately.
            self._load_weights(model)

            # Use the Adam update rule.
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            logging.info('Starting compilation...')
            model.compile(loss='categorical_crossentropy', optimizer=adam)
            logging.info('Finished compilation.')

            if self.params['predict']:
                self._display_wrong_predictions(model, X_val, y_val)
            else:
                self._evaluate(model, X_val, Y_val)

        else:
            # Train and evaluate model.

            # Use the Adam update rule.
            opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            logging.info('Starting compilation...')
            
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt, 
                          metrics=['accuracy'])
            
            logging.info('Finished compilation.')

            # Settings for preprocessing.
            datagen = ImageDataGenerator(
                featurewise_center=True,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=True,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,
                data_format='channels_first')  # randomly flip images

            datagen.fit(X_train)

            '''
            # Fit the model on the batches generated by datagen.flow().
            history = History()
            callbacks = [history]
            '''

            if save_weights:
                file_name = self._get_file_prefix() + '.hdf5'
                checkpointer = ModelCheckpoint(filepath=file_name, save_best_only=True, mode='auto', verbose=1, monitor="val_acc")
                callbacks.append(checkpointer)
            
            print(X_train.shape)
            print(X_val.shape)
            model.summary()
            
            
            
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                          epochs=nb_epoch,
                                          steps_per_epoch=X_train.shape[0] // batch_size,
                                          validation_data=(X_val, Y_val),
                                          validation_steps = X_val.shape[0] // batch_size)
                                          
            '''
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                          samples_per_epoch=X_train.shape[0],
                                          nb_epoch=nb_epoch,
                                          validation_data=(X_val, Y_val),
                                          nb_worker=1, callbacks=callbacks, verbose=2)'''

            
            print(history.history.keys())
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            plt.savefig('acc_cnn_stan_orig_333.png')
            plt.close()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            plt.savefig('loss_cnn_stan_orig_333.png')
            plt.close()
            
            '''# Print the results to the console
            for key in history.history:
                print(key, history.history[key])

            final_acc = history.history["acc"][-1] 

            # Print the results to a file
            out_file = self._get_file_prefix() + '_' + str(final_acc) + "_out.txt"
            f = open(out_file, "w")
            for key in history.history:
                f.write(key + ": " + str(history.history[key]) + "\n")

            # print parameters to the file
            for key in self.params:
                f.write(key + ": " + str(self.params[key]) + "\n")

            f.close()'''


# In[ ]:




# In[119]:

cnn = CNN()

cnn.load_data(train_data_file_default, val_data_file_default, num_train=num_train, num_val=num_val)


# In[120]:

cnn.train()


# In[ ]:




# In[ ]:



