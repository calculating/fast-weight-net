import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class MetaNet(layers.Layer):
    # simple convolutional. takes sample, outputs activations for each layer
    def __init__(self):
        print("MetaNet init")
        super(MetaNet, self).__init__(trainable=False)
        self.conv_1a = layers.Conv2D(10, 3, activation='relu')
        self.conv_1b = layers.Conv2D(10, 3, activation='relu')
        self.conv_2a = layers.Conv2D(10, 3, activation='relu')
        self.conv_2b = layers.Conv2D(10, 3, activation='relu')
        self.denseout = layers.Dense(10, activation='softmax')
        self.pooler = layers.MaxPool2D([2,2])
        self.flatten = layers.Flatten()

    def call(self, input):
        activations = []

        x = self.conv_1a(input)
        print(x)
        activations.append([tf.math.reduce_mean(x, axis=[1,2]), tf.math.reduce_std(x, axis=[1,2])])
        print('---------')

        x = self.conv_1b(x)
        print(x)
        activations.append([tf.math.reduce_mean(x, axis=[1,2]), tf.math.reduce_std(x, axis=[1,2])])

        x = self.pooler(x)
        print(x)

        x = self.conv_2a(x)
        activations.append([tf.math.reduce_mean(x, axis=[1,2]), tf.math.reduce_std(x, axis=[1,2])])

        x = self.conv_2b(x)
        activations.append([tf.math.reduce_mean(x, axis=[1,2]), tf.math.reduce_std(x, axis=[1,2])])

        x = self.pooler(x)
        x = self.flatten(x)
        
        x = self.denseout(x)

        return activations, x

class Modulator(layers.Layer):
    # takes compressed activation data, hidden state
    # returns embedding map, hidden state
    def __init__(self, map):
        print("Modulator init")
        super(Modulator, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(50, activation='relu')
        self.d2 = layers.Dense(map*3, activation='relu')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.d1(x)
        return self.d2(x)
    
class ConvWeigher(layers.Layer):
    # takes stacked embeddings, expanded activation data
    # returns adjusted weights
    def __init__(self):
        print("Conv Weigher init")
        super(ConvWeigher, self).__init__()
        self.d1 = layers.Dense(10, activation='relu')
        self.d2 = layers.Dense(9, activation='relu')
    
    def call(self, inputs):
        x = tf.map_fn(self.d1, inputs)
        return tf.map_fn(self.d2, x)

class DenseWeigher(layers.Layer):
    # takes stacked embeddings, expanded activation data
    # returns adjusted weights
    def __init__(self):
        print("Dense Weigher init")
        super(DenseWeigher, self).__init__()
        self.d1 = layers.Dense(10, activation='relu')
        self.d2 = layers.Dense(10, activation='relu')

    def call(self, inputs):
        x = tf.fn_map(self.d1, inputs)
        return tf.fn_map(self.d2, x)
    
class TrainerCell(layers.Layer):
    # MetaNet followed by Modulator followed by Weigher
    # takes a single input sample, hidden state
    # returns weights, hidden state
    def __init__(self):
        print("Trainer init")
        super(TrainerCell, self).__init__()

        self.MetaNet = MetaNet()

        self.state_size = 100
        self.output_size = 17490
        mapSize = 50
        self.Modulator = Modulator(mapSize)
        self.ConvLocal = ConvWeigher()
        self.DenseLocal = DenseWeigher()
        
    def call(self, inputs, state):
        print("trainer input init, -------------", inputs, state)
        # pixel values and hidden state
        x = inputs
        h = state
        # get convolution filter activations and output activations from the MetaNet
        act, out = self.MetaNet(x)

        # weight change embedding maps for the convolution and dense layers, new hidden state
        metaInf = tf.concat((act, out, h), axis=-1)
        embwC, embwD, hNew = self.Modulator(metaInf)
        # stack the embeddings
        embwC = tf.stack(embwC, axis=1)
        embwD = tf.stack(embwD, axis=1)
        # concatenate each embedding with the corresponding activation data
        convInfo = tf.concat([act, embwC], axis=2)
        denseInfo = tf.concat([out, embwD], axis=2)

        # get the new conv filter weights
        weights = self.ConvLocal(convInfo)
        # add the new dense layer weights
        weights.append(self.DenseLocal(denseInfo))
        # apply the new weights to the MetaNet
        self.MetaNet.set_weights(weights)
        # return the weights for testing, and the new hidden state
        return weights, hNew

class Predictor(layers.Layer):
    # takes weights, test sequence
    # returns predictions
    def __init__(self):
        print("Predictor init")
        super(Predictor, self).__init__(trainable=False)
        self.TestNet = MetaNet()
    def call(self, inputs):
        weights, x = inputs
        self.TestNet.set_weights(weights)
        return self.TestNet(x)

class MetaLearner(tf.keras.Model):
    # Trainer followed by Predictor
    # takes sample sequence and test sequence
    # returns predictions
    def __init__(self):
        print("MetaLearner init")
        super(MetaLearner, self).__init__()
        self.Trainer = layers.RNN(TrainerCell(), unroll=True)
        self.Predictor = Predictor()
    def call(self, inputs):
        print("MetaLearner call", tf.shape(inputs))
        startState = tf.keras.layers.Input(shape=(100,))
        trainedWeights = self.Trainer(inputs, initial_state=startState)
        return self.Predictor([trainedWeights, inputs])



amyg = MetaLearner()

amyg.call(tf.zeros([10, 10, 64, 64, 3]))

amyg.summary()