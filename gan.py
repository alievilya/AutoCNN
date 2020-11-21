import random
from typing import Dict, Callable

import numpy as np
import tensorflow as tf

from cnn_structure import SkipLayer, PoolingLayer, CNN

random.seed(42)


class AutoCNN:
    def get_input_shape(self):
        return (32, 32, 3)

    def __init__(self, population_size: int, maximal_generation_number: int, dataset: Dict[str, np.ndarray],
                 output_layer: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer]):
        self.output_layer = output_layer
        self.dataset = dataset

        self.maximal_generation_number = maximal_generation_number
        self.population_size = population_size
        self.population = []

        self.input_shape = self.get_input_shape()

        self.initialize()

    def initialize(self):
        self.population.clear()

        for _ in range(self.population_size):
            depth = random.randint(1, 10)
            cnn = CNN(self.input_shape, self.output_layer)

            for i in range(depth):
                r = random.random()

                if r < .5:
                    f1 = 2 ** random.randint(5, 9)
                    f2 = 2 ** random.randint(5, 9)

                    cnn.append(SkipLayer(f1, f2))

                else:
                    q = random.random()

                    if q < .5:
                        cnn.append(PoolingLayer('max'))
                    else:
                        cnn.append(PoolingLayer('mean'))

            self.population.append(cnn)


if __name__ == '__main__':
    def output_function(inputs):
        out = tf.keras.layers.Flatten()(inputs)

        return tf.keras.layers.Dense(10, activation='softmax')(out)


    a = AutoCNN(10, 3, None, output_function)

    print(a.population)
    a.population[0].generate()
