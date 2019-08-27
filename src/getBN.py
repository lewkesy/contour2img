import os
import importlib, argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import data
import util
from model import *

tfile = "../data/output.npy"
model_path = "../wts/train/0.0002_1_0.3/"
data_path = '../data/'
trainset_path = data_path + 'trainset.txt'
testset_path  = data_path + 'testset.txt'

niter = 100

imgs_color = np.load(tfile)
batch_size = 36

# imgs_color = imgs_color[0]
# imgs_color = np.transpose(imgs_color, (1, 2, 0))
# imgs_color = np.array(imgs_color[np.newaxis, :, :, :])
# embed()

# load data

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


saver = util.ckpter(model_path + 'model*.npz')
mfile = saver.latest

d = data.dataset(batch_size, imgs_color.shape[1], imgs_color.shape[2], 0, 0, 0, 0.3)
model = Model()

graph_input = d.imgs_color
x1_stage = model.build_LFN(graph_input, reuse=tf.AUTO_REUSE)
x2_stage = model.build_HFN(graph_input, x1_stage, reuse=tf.AUTO_REUSE)

nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())

wts = np.load(mfile)
ph = tf.placeholder(tf.float32)
for k in model.weights.keys():
    wvar = model.weights[k]
    wk = wts[k].reshape(wvar.get_shape())
    sess.run(wvar.assign(ph),feed_dict={ph: wk})


bnnms, bnwts = list(model.bnwts.keys()), list(model.bnwts.values())
bns = {k: [] for k in bnnms}

train_batch_nums = len(trainset) // batch_size
rs = np.random.RandomState(0)
idx = 0

for i in range(niter):
    print(i)
    if i % train_batch_nums == 0:
        idx = rs.permutation(len(trainset))
    image_paths = [trainset[idx[(iters % train_batch_nums) * batch_size + b]] for b in range(batch_size)]
    # TODO Debug here
    # sess.run(d.imgs_color, feed_dict=d.dict(imgs_color))
    # sess.run(bnwts)
    _, tbns = sess.run([x2_stage, bnwts], feed_dict=d.dict(image_paths, 1, 1, True))
    for j in range(len(bnnms)):
        nm = bnnms[j]
        bns[nm].append(tbns[j])

# average over all batches
for nm in bnnms:
    if '_v' in nm:
        continue
    vnm = nm.replace('_mu', '_v')

    mean = np.mean(np.stack(bns[nm], axis=0), axis=0)
    mean_var = np.var(np.stack(bns[nm], axis=0), axis=0)
    var = np.mean(np.stack(bns[vnm], axis=0), axis=0)

    bns[nm] = mean
    bns[vnm] = mean_var + var

# Save population stat
ofile = mfile.replace('model','bnwts')
# embed()
np.savez(ofile, **bns)