"""
    MobileNet-v1 model written in TensorFlow Keras
"""
from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, \
    DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

def MobileNetv1():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool'))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def remove_channel(model):
    '''
    Input: model
           description: which is the model to be indeed pruned 
    Ouput: new_model
           description: which is the new model generating by removing all-zero channels
    '''
    def create_model(model):
        new_model = Sequential()
        score_list = np.sum(np.abs(model.layers[0].get_weights()[0]), axis=(0,1,2))
        next_layer_score_list = np.sum(np.abs(model.layers[0+3].get_weights()[0]), axis=(0,1,3))
        score_list = score_list * next_layer_score_list
        out_planes_num = int(np.count_nonzero(score_list))

        new_model.add(Conv2D(out_planes_num, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))

        new_model.add(DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))

        for i in range(6,75,6):
            if i == 78:
                pass
            else:
                if isinstance(model.layers[i],Conv2D):
                    old_strides = model.layers[i+3].strides
                    score_list = np.sum(np.abs(model.layers[i].get_weights()[0]), axis=(0,1,2))
                    next_layer_score_list = np.sum(np.abs(model.layers[i+3].get_weights()[0]), axis=(0,1,3))
                    score_list = score_list * next_layer_score_list
                    out_planes_num = int(np.count_nonzero(score_list))
                    out_planes_idx = np.squeeze( np.nonzero(score_list))
                            
                    new_model.add(Conv2D(out_planes_num, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
                    new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
                    new_model.add(Activation('relu'))

                    new_model.add(DepthwiseConv2D((3, 3), strides=old_strides, padding='same', use_bias=False))
                    new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
                    new_model.add(Activation('relu'))

        score_list = np.sum(np.abs(model.layers[78].kernel), axis=(0,1,2))
        out_planes_num = int(np.count_nonzero(score_list))

        new_model.add(Conv2D(out_planes_num, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False))
        new_model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
        new_model.add(Activation('relu'))
        new_model.add(AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool'))
        new_model.add(Flatten())
        new_model.add(Dense(10))
        new_model.add(Activation('softmax'))
        return new_model

    new_model = create_model(model)

    def update_model(new_model, model):
        score_list = np.sum(np.abs(model.layers[0].get_weights()[0]), axis=(0,1,2))
        next_layer_score_list = np.sum(np.abs(model.layers[0+3].get_weights()[0]), axis=(0,1,3))
        score_list = score_list * next_layer_score_list
        out_planes_idx = np.squeeze( np.nonzero(score_list))
        old_wgt=model.layers[0].get_weights()[0]

        new_model.layers[0].set_weights([old_wgt[:,:,:,out_planes_idx]])
        old_wgt=model.layers[3].get_weights()[0]
        new_model.layers[3].set_weights([old_wgt[:,:,out_planes_idx,:]])
        input_planes_index = out_planes_idx
        for i in range(6,75,6):
            if i == 78:
                pass
            else:
                if isinstance(model.layers[i],Conv2D):
                    old_strides = model.layers[i+3].strides
                    score_list = np.sum(np.abs(model.layers[i].get_weights()[0]), axis=(0,1,2))
                    next_layer_score_list = np.sum(np.abs(model.layers[i+3].get_weights()[0]), axis=(0,1,3))
                    score_list = score_list * next_layer_score_list
                    out_planes_idx = np.squeeze( np.nonzero(score_list))

                    old_wgt=model.layers[i].get_weights()[0]
                    new_model_weigths=new_model.layers[i].get_weights()[0]
                    for idx,idx_out in enumerate(out_planes_idx):
                        new_model_weigths[:,:,:, idx] = old_wgt[:,:,input_planes_index,idx_out]
                    new_model.layers[i].set_weights([new_model_weigths])
                    old_wgt=model.layers[i+3].get_weights()[0]
                    new_model.layers[i+3].set_weights([old_wgt[:,:,out_planes_idx,:]])
                    input_planes_index = out_planes_idx
        score_list = np.sum(np.abs(model.layers[78].get_weights()[0]), axis=(0,1,2))
        out_planes_idx = np.squeeze( np.nonzero(score_list))


        new_model_weigths = new_model.layers[78].get_weights()[0]
        old_wgt=model.layers[78].get_weights()[0]
        for idx, idx_out in enumerate(out_planes_idx):
            new_model_weigths[:, :, :, idx] = old_wgt[:, :, input_planes_index, idx_out]
        new_model.layers[78].set_weights([new_model_weigths])

        old_wgt=model.layers[83].get_weights()
        new_model.layers[83].set_weights([old_wgt[0][out_planes_idx,:], old_wgt[1]])

        return new_model
    new_model=update_model(new_model,model)
    return new_model

def channel_fraction_pruning(model, fraction):
    for layer in model.layers:
        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            weights = np.array(layer.get_weights())
            l1_norms = [np.sum(np.abs(weights[:,:,:,:,i])) for i in range(weights.shape[4])]
            num_pruned = int(len(l1_norms)*fraction)
            prune_filter_indices = np.argpartition(l1_norms, min(len(l1_norms)-1, num_pruned))
            for i in range(num_pruned):
                weights[:,:,:,:,prune_filter_indices[i]].fill(0)
            layer.set_weights(weights)
    return model

def print_weights_per_layer(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            weights = np.array(layer.get_weights())
            print(weights.shape)
            for i in range(weights.shape[4]):
                print(weights[:,:,:,:,i])
    return model

global train_norm
def convert_tflite(model, name="", optim=False):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optim:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Static quantization
        # converter.representative_dataset = representative_data
    tflite_model = converter.convert()

    # Save the model
    with open(f"models/{name}.tflite",'wb') as f:
        f.write(tflite_model)


def representative_data():
    data = tf.data.Dataset.from_tensor_slices(train_norm).batch(1).take(100)
    for input_value in data:
        yield [input_value]

if __name__ == '__main__':
    # fractions = [0.2, 0.5, 0.9]
    fractions = [0.2]
    fine_tuning_epochs=[5]
    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Convert the datasets to contain only float values
    train_norm = X_train.astype('float32')
    test_norm = X_test.astype('float32')

    # Normalize the datasets
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # Encode the labels into one-hot format
    trainY = to_categorical(y_train)
    testY = to_categorical(y_test)

    # load pretrained MobileNetv1
    model = MobileNetv1()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(train_norm, trainY, epochs=20, batch_size=128, validation_data=(test_norm, testY), verbose=1)
    # print(str(model.evaluate(x=test_norm, y=testY, verbose=1)) + ' Trainable Params = ' + str(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])), flush=True)
    # model.save_weights('mbnv1_tf.ckpt')
    load_status = model.load_weights("mbnv1_tf.ckpt")
    print(str(model.evaluate(x=test_norm, y=testY, verbose=1)))
    for f in fractions:
        for fte in fine_tuning_epochs:
            clone_model = tf.keras.models.clone_model(model)
            clone_model.build((1, 32, 32, 3))
            clone_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            clone_model.set_weights(model.get_weights())
            #prune model
            channel_pruned_model = channel_fraction_pruning(clone_model, f)  
            # remove channels
            new_model = remove_channel(channel_pruned_model)

            new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = new_model.fit(train_norm, trainY, epochs=fte,
                    batch_size=128, validation_data=(test_norm, testY), verbose=1)
            print(str(f) + ' ' + str(fte) + ' ' + str(new_model.evaluate(x=test_norm, y=testY,verbose=1)) + ' Trainable Params = ' + str(np.sum([np.prod(v.get_shape()) for v in new_model.trainable_weights])), flush=True)

            # convert_tflite(new_model, name=f"mbnv1_frac{f}_ep{fte}", optim=False)
            convert_tflite(new_model, name=f"mbnv1_frac{f}_ep{fte}_optim", optim=True)

