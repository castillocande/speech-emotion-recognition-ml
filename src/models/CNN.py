import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


class CNN:
    """Clase para definir, entrenar y evaluar el modelo CNN."""

    def __init__(self, name, input_dim=(224, 224, 3), learning_rate=0.0001, momentum=80, epochs=50, batch_size=32):
        self.name = name
        self.img_width = input_dim[0]
        self.img_height = input_dim[1]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.create_model((input_dim[0], input_dim[1], 3), 8)

    def create_model(self, input_shape, num_classes):
        """Crea el modelo CNN."""
        inputs = Input(shape=input_shape)
        x = Conv2D(8, (3, 3), activation="relu")(inputs)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs, outputs)
        return model

    def compile_model(self): 
        """Compila el modelo."""
        optimizer = RMSprop(learning_rate=self.learning_rate, momentum=self.momentum)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, train_dir):
        """Entrena el modelo usando los datos de entrenamiento y validación."""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.125,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_generator = tf.keras.preprocessing.image.DirectoryIterator(
            directory=train_dir,
            image_data_generator=train_datagen,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // self.batch_size,
            epochs=self.epochs
        )

    def evaluate(self, test_dir):
        """Evalúa el modelo usando los datos de prueba."""
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        test_loss, test_accuracy = self.model.evaluate(test_generator, steps=test_generator.samples // self.batch_size)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    def plot_learning_curves(self):
        """Grafica las curvas de aprendizaje del entrenamiento y validación."""
        history_dict = self.history.history
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]
        accuracy = history_dict["accuracy"]
        val_accuracy = history_dict["val_accuracy"]

        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.show()
