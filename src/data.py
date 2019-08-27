import numpy as np
import tensorflow as tf
from IPython import embed
from util import edge_detector, mask_fill
class dataset:
    def __init__(self, batch_size, height=256, width=256, c_height=128, c_width=128, overlap=16, edge_scale=0.2):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.c_height = c_height
        self.c_width = c_width
        self.overlap = overlap
        self.edge_scale = edge_scale
        self.graph()

    def graph(self):
        self.ids = []
        self.is_train = tf.placeholder(tf.bool, [])
        self.gen_learning_rate = tf.placeholder( tf.float32, [])
        self.dis_learning_rate = tf.placeholder( tf.float32, [])
        for i in range(self.batch_size):
            self.ids.append(tf.placeholder(tf.string))
        
        self.imgs = []
        for i in range(self.batch_size):
            img = tf.image.decode_jpeg(tf.read_file(self.ids[i]), channels=3) # 

            # crop from origin datasets
            in_s = tf.to_float(tf.shape(img)[:2])
            min_s = tf.minimum(in_s[0], in_s[1])
            new_s = tf.to_int32((float(self.height + 1) / min_s) * in_s)
            img = tf.image.resize_images(img, [new_s[0], new_s[1]]) 
            img = tf.random_crop(img, [self.height, self.width, 3])

            self.imgs.append(img)

        self.imgs = tf.stack(self.imgs)
        self.precontour = tf.identity(self.imgs)
        self.imgs = tf.to_float(tf.stack(self.imgs))*(2.0/255.0) - 1.0

        self.precontour = edge_detector(tf.image.rgb_to_grayscale(self.precontour), maxRate = self.edge_scale)
        
        self.contour, self.sparse_points = mask_fill(self.precontour, k = 30)
        self.contour = tf.to_float(tf.greater_equal(self.contour, 1))
        # self.contour = tf.expand_dims(self.contour, axis=3)

        imgs_gradient_y = tf.concat([tf.expand_dims(self.imgs[:, -1, :, :], 1), self.imgs[:, :-1, :, :]], axis=1) - self.imgs
        imgs_gradient_x = tf.concat([tf.expand_dims(self.imgs[:, :, -1, :], 2), self.imgs[:, :, :-1, :]], axis=2) - self.imgs
        self.imgs_gradient = tf.concat([imgs_gradient_y, imgs_gradient_x], axis=3) * self.contour
        self.sparse_gradient = tf.concat([imgs_gradient_y, imgs_gradient_x], axis=3) * self.sparse_points
        
        img_d = tf.concat([tf.expand_dims(self.imgs[:, -1, :, :], 1), self.imgs[:, :-1, :, :]], axis=1)
        img_b = tf.concat([self.imgs[:, 1:, :, :], tf.expand_dims(self.imgs[:, 0, :, :], 1)], axis=1)
        self.imgs_color = tf.concat([img_d, img_b], axis=3) * self.contour
        self.pre_imgs_color = tf.concat([img_d, img_b], axis=3) * self.precontour
        self.sparse_color = tf.concat([img_d, img_b], axis=3) * self.sparse_points
        

    def dict(self, ids, gen_learning_rate, dis_learning_rate, is_train):
        fd = {}
        fd[self.gen_learning_rate] = gen_learning_rate
        fd[self.dis_learning_rate] = dis_learning_rate
        fd[self.is_train] = is_train
        for i in range(self.batch_size):
            fd[self.ids[i]] = ids[i]

        return fd


