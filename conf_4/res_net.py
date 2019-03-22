# This version of code uses more data set augumentation

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Params
DIMENSION = 224  # Will generate images DIMENSIONxDIMENSION
BATCH_SIZE = 32
EPOCHS = 250
LR = [0.0001]
L2_REG = 0.001

FOLDER = 'res_net_aug_reg_l2_16'
LOAD_MODEL = False
MODEL = '/home/nct01/nct01075/projects/SDOGS/' + FOLDER + '/model_62'
TRAIN_DATA_PATH = '/home/nct01/nct01075/projects/SDOGS/SDOGS-preproc'
VALID_DATA_PATH = '/home/nct01/nct01075/projects/SDOGS/SDOGS-preproc-test'
CHECKPOINT_PATH = '/home/nct01/nct01075/projects/SDOGS/' + FOLDER + '/checkpoints/model.{epoch:002d}-{val_acc:.2f}'
PLOT_PREFIX = 'res_net50'
CLASSES = 120
VALID_SIZE = 8580
IMAGES = 20580 - VALID_SIZE


def get_model(input_shape, output_shape, optimizer, reg):
    # Create input
    image_input = Input(shape=input_shape, name='image_input')

    # Get pretrained model
    modelResNet = ResNet50(
        weights='imagenet',
        include_top=False
    )

    # We train only FC layers
    modelResNet.trainable = False

    # Use the generated model
    outputResNet = modelResNet(image_input)

    # Add top layers to pretrained model
    top = Flatten(name='flatten')(outputResNet)
    top = Dense(1024, activation='relu', name='fc2',
                kernel_regularizer=reg)(top)
    top = Dropout(0.5)(top)
    top = Dense(output_shape, activation='softmax', name='predictions')(top)

    # Create, compile and train model
    model = Model(inputs=image_input, outputs=top)
    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


# Define data source
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,
    batch_size=BATCH_SIZE,
    target_size=(DIMENSION, DIMENSION),
    class_mode='categorical',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    VALID_DATA_PATH,
    batch_size=BATCH_SIZE,
    target_size=(DIMENSION, DIMENSION),
    class_mode='categorical',
    shuffle=True
)

for lr in LR:
    # Create checkpoint callback
    modelCheckpoint = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True)

    # Get model
    model = get_model(
        input_shape=(DIMENSION, DIMENSION, 3),
        output_shape=CLASSES,
        optimizer=Adam(lr=lr),
        reg=l2(L2_REG)
    )

    if LOAD_MODEL:
        model.load_weights(MODEL)

    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=VALID_SIZE,
        epochs=EPOCHS,
        steps_per_epoch=IMAGES//BATCH_SIZE,
        callbacks=[modelCheckpoint]
    )

    # Store Plots
    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(PLOT_PREFIX + '_accuracy.pdf')
    plt.close()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(PLOT_PREFIX + '_loss.pdf')
    plt.close()
