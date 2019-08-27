
import os
from glob import glob
import pandas as pd
import numpy as np
import skimage.io
from tmodel import *
# from model import *
import util
from IPython import embed
import time
import data_test as data
import argparse
import cv2

parser = argparse.ArgumentParser(description="Learning rate and lambda recon")
parser.add_argument('--lr', type=float, default=0.0002, required=False)
parser.add_argument('--pre', type=int, default=1, required=False)
parser.add_argument('--t', type=int, default=1, required=False)
parser.add_argument('--s', type=float, default=0.3, required=False)

args = parser.parse_args()

# contour_path = "../data/unet_res.npy"

ind = 4
contour_path = "../data/contour_0_1.4_0.5_0.0001_0.01_10.npy"
contour_path = "../data/grid_contour_0_1.6_0.5_1e-05_0.5_15.npy"
# contour_path = "../data/test.npy"
ori_path = "../data/unet_ori.npy"
model_path = "../wts/train/{}_{}_{}/".format(args.lr, args.t, args.s)
result_path = "../results/test/{}_{}_{}/".format(args.lr, args.t, args.s)

if not os.path.exists(result_path):
    os.makedirs(result_path)

imgs_color = np.load(contour_path)
# imgs_color = imgs_color[0]
# imgs_color = np.transpose(imgs_color, (1,2,0))
# imgs_color = imgs_color[np.newaxis, :, :, :]
# imgs_color = [imgs_color for i in range(12)]
imgs_color = np.array(imgs_color)
ori_imgs = np.load(ori_path)
# embed()
saver = util.ckpter(model_path + 'model*.npz')
batch_size = imgs_color.shape[0]

tfile = saver.latest
tbfile = tfile.replace('model','bnwts')

# d = data.dataset(1, imgs_color.shape[1], imgs_color.shape[2])
d = data.dataset(1, imgs_color.shape[1], imgs_color.shape[2])

model = Model()

bnwts = {}
wts = np.load(tbfile)
for bnnm in wts.keys():
    bnwts[bnnm] = tf.Variable(tf.random_uniform(wts[bnnm].shape),trainable=False)
model.bnwts = bnwts

if args.t == 2:
    graph_input = d.contour
    graph_input = model.build_contour_representation(graph_input)
elif args.t == 0:
    graph_input = d.imgs_gradient
elif args.t == 1:
    graph_input = d.imgs_color

x1_stage = model.build_LFN(graph_input, reuse=tf.AUTO_REUSE)
if args.pre == 1:
    x2_stage = model.build_HFN(graph_input, x1_stage, reuse=tf.AUTO_REUSE)


sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess.run(tf.global_variables_initializer())

wts = np.load(tfile)
ph = tf.placeholder(tf.float32)
for k in wts.keys():
    if k in model.weights:
        wvar = model.weights[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(ph),feed_dict={ph: wk})

wts = np.load(tbfile)
for k in wts.keys():
    if k in model.bnwts:
        wvar = model.bnwts[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(ph),feed_dict={ph: wk})
# embed()
# imgs_color = [imgs_color[0,:,:,:] for _ in range(batch_size)]
# imgs_color = np.transpose(imgs_color, (0, 2, 3, 1 ))
# embed()

# embed()
# imgs_color = np.transpose(imgs_color, (1, 2, 0))
# imgs_color = imgs_color[:256, :256, :]
# embed()
# cv2.imwrite('test.png', ((imgs_color[:, :, 0] + 1) * 255. / 2).astype(int))
cv2.imwrite('test.png', ((imgs_color[0, :, :, 0] + 1) * 255. / 2).astype(int))
# imgs_color = imgs_color[None, :, :, :]
for i in range(10):
    output = sess.run(x2_stage, feed_dict=d.dict(np.expand_dims(imgs_color[i], axis=0)))
    # output[0, 64:64+128,64:64+128, :] = ori_imgs[i, 64:64+128,64:64+128, :]
    output_img = (255 * (1 + output[0]) / 2).astype(int)
    skimage.io.imsave( os.path.join(result_path, 'ori_'+str(i) + '.jpg'), (255 * (1 + ori_imgs[i]) / 2).astype(int))
    skimage.io.imsave( os.path.join(result_path, 'res_'+str(i) + '.jpg'), output_img)
# embed()

# cv2.imwrite( os.path.join(result_path, 'res_'+str(0) + '.jpg'), output_img)