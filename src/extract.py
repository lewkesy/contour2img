
import os
from glob import glob
import pandas as pd
import numpy as np
import skimage.io
from model import *
import util
from IPython import embed
import time
import data
import argparse

parser = argparse.ArgumentParser(description="Learning rate and lambda recon")
parser.add_argument('--lr', type=float, default=0.0002, required=False)
parser.add_argument('--pre', type=int, default=1, required=False)
parser.add_argument('--t', type=int, default=1, required=False)
parser.add_argument('--s', type=float, default=0.3, required=False)
args = parser.parse_args()

batch_size = 400
if args.pre == 1:
    batch_size = batch_size // 2
attention_gamma = 1.0 / 40

coarse_l1_alpha = 1.2
l1_loss_alpha = 1.2
ae_loss_alpha = 1.2
gan_loss_alpha = 0.01
global_loss_alpha = 1
gan_gp_lambda = 10

# test
d_overlap_loss_alpha = 0.2
# g_overlap_loss_alpha = args.o
# loss_gradient_alpha = args.g
overlap_patch_num = 8

overlap_size = 32
img_size = 256
hiding_size = 128
contour_scale = args.s

gen_learning_rate_val = args.lr
dis_learning_rate_val = args.lr / 10

SAVEITER = 1e4
DISPITER = 10
VALITER = 1000
VALREP = 2
MAXITER = 50e3

data_path = '../data/'
trainset_path = data_path + 'trainset.txt'
testset_path  = data_path + 'testset.txt'
# dataset_path = '/scratch/data/Places365/data_large/a/'
dataset_path = 'D:/deep learning/StreetView/'
mission_path = str(args.lr) + "_" + str(args.t) + "_" + str(args.s) + '/'
model_path = '../wts/train/' + mission_path
result_path= '../results/train/' + mission_path
plot_path = '../plot/train/' + mission_path


if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

saver = util.ckpter(model_path + 'model*.npz')
if saver.iter >= MAXITER:
    MAXITER=350e3
    # learning_rate_val = 1e-4

####################### data organization ####################################

if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):
    input_data = []
    for root, dirs, _, in os.walk(dataset_path):
        input_data.extend( glob( os.path.join(root, '*.jpg')))
        for dir in dirs:
            subdir_path = os.path.join(dataset_path, dir)
            for root, _, _ in os.walk(subdir_path):
                input_data.extend( glob( os.path.join(root, '*.jpg')))


    input_data = np.hstack(input_data)
    trainset = input_data[:int(len(input_data)*0.9)]
    testset = input_data[int(len(input_data)*0.9):]
    # trainset = pd.DataFrame({'image_path':input_data[:int(len(input_data)*0.9)]})
    # testset = pd.DataFrame({'image_path':input_data[int(len(input_data)*0.9):]})
    with open(trainset_path, "w") as f:
        for line in trainset:
            f.write(line + "\n")

    with open(testset_path, "w") as f:
        for line in testset:
            f.write(line + "\n")
    # trainset.to_pickle( trainset_path )
    # testset.to_pickle( testset_path )
else:
    with open(trainset_path, "r") as f:
        trainset = [l.rstrip('\n') for l in f.readlines()]
    
    with open(testset_path, "r") as f:
        testset = [l.rstrip('\n') for l in f.readlines()]

labels_D = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], 0)
labels_G = tf.ones([batch_size])

# masked region
################### image_hiding is replaced by d.crop_imgs, image_tf is relpaced by d.imgs

# images_hiding = tf.placeholder( tf.float32, [batch_size, hiding_size, hiding_size, 3], name='images_hiding')
# images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")
d = data.dataset(batch_size, img_size, img_size, hiding_size, hiding_size, overlap_size, contour_scale)

model = Model()

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
imgs_pos = d.imgs

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())

tf.global_variables_initializer().run()

#TODO Load net seems not work
origiter = saver.iter

train_batch_nums = len(trainset) // batch_size
test_batch_nums = len(testset) // batch_size
iters = origiter

if origiter > 0:
    util.loadNet(saver.latest,model,sess)
    if os.path.isfile('../wts/train/' + mission_path + 'popt.npz'):
        util.loadAdam('../wts/train/' + mission_path + 'popt.npz',opt,model.weights,sess)
    util.mprint("Restored to iteration %d" % origiter)    


test_image_paths = testset[:batch_size]
contour, contour_imgs, ori_imgs, sparse_colors = sess.run(
        [d.precontour, d.imgs_color, imgs_pos, d.sparse_color],
        feed_dict=d.dict(test_image_paths, gen_learning_rate_val, dis_learning_rate_val, False))

np.save("../results/output.npy", contour_imgs)
np.save("../results/sparse.npy", sparse_colors)
np.save("../results/ori.npy", ori_imgs)
np.save("../results/contour.npy", contour[:, :, :, 0])

for i in range(5):
    output_img = (255 * (1 + ori_imgs[i]) / 2).astype(int)
    skimage.io.imsave( os.path.join(result_path, 'ori_'+str(i)+'.jpg'), output_img)
    # embed()
    output_img = (255 * (contour[i, :, :, 0])).astype(int)
    skimage.io.imsave( os.path.join(result_path, 'con_'+str(i)+'.jpg'), output_img)