from abc import ABC, abstractmethod
import tensorflow as tf

INPUT_NAME = 'input'
HIDDEN_NAME = 'hidden'
OUTPUT_NAME = 'output'


class MLPBase(ABC):
    def __init__(self, dimensions):
        pass

    def build_network(self):
        pass

    @abstractmethod
    def train(self):
        pass
