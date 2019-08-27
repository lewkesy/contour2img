import numpy as np
import tensorflow as tf
from IPython import embed
from util import edge_detector
class dataset:
    def __init__(self, batch_size, height=256, width=256):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.graph()

    def graph(self):
        self.imgs_color = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, 6])

    def dict(self, imgs_color):
        fd = {}
        fd[self.imgs_color] = imgs_color


        return fd


