import tensorflow as tf
import argparse
from models.vgg_tf import VGG
from models.mobilenet_tf import MobileNetv1
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW4 - Starter TensorFlow code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
parser.add_argument('--model_type', type=str, default='VGG11', help='Model type')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

random_seed = 1
tf.random.set_seed(random_seed)

# TODO: Insert your model here
if args.model_type == 'VGG11':
    model = VGG('VGG11')
elif args.model_type == 'VGG16':
    model = VGG('VGG16')
else:
    model = MobileNetv1()
model.summary()

# TODO: Load the training and testing datasets
(trainX, trainy), (testX, testy) = cifar10.load_data()

# TODO: Convert the datasets to contain only float values
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')

# TODO: Normalize the datasets
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

# TODO: Encode the labels into one-hot format
trainY = to_categorical(trainy)
testY = to_categorical(testy) 

# TODO: Configures the model for training using compile method
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: Train the model using fit method
start = time.time()
history = model.fit(train_norm, trainY, epochs=epochs, batch_size=batch_size, validation_data=(test_norm, testY), verbose=1)
training_time = time.time() - start

# TODO: Save the weights of the model in .ckpt format
model.save_weights(args.model_type + '.ckpt')

with open(args.model_type + ".csv", 'w+') as file:
    dict = history.history
    file.write(f"Training time: {training_time},,,,\n")
    file.write("Epoch,Train loss,Train accuracy,Test loss,Test accuracy\n")
    for epoch in range(epochs):
        file.write(f"{epoch},{dict['loss'][epoch]},{dict['accuracy'][epoch]},{dict['val_loss'][epoch]},{dict['val_accuracy'][epoch]}\n")

print("total training time : ", training_time)
