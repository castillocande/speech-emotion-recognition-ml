import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


class CNN:
    def __init__(self, type, name, train_dir, test_dir, img_width=224, img_height=224, batch_size=32, seed_value=32, learning_rate=0.0001, momentum=0.8, epochs=50):
        self.type = type
        self.name = name
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.seed_value = seed_value
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self._prepare_data_generators()
        
    def _prepare_data_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.125,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        self.val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
    def create_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)
        
        x = Conv2D(8, (3, 3), activation='relu')(inputs)
        x = AveragePooling2D((2, 2))(x)
        
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = AveragePooling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = AveragePooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)  
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)  
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
    def compile_model(self):
        optimizer = RMSprop(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
    def train_model(self):
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            epochs=self.epochs
        )
        
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, steps=self.test_generator.samples // self.batch_size)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        