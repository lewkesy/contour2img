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
parser.add_argument('--lr', type=float, default=0.0001, required=False)
parser.add_argument('--pre', type=int, default=0, required=False)
parser.add_argument('--t', type=int, default=0, required=False)
parser.add_argument('--s', type=float, default=0.2, required=False)

args = parser.parse_args()

batch_size = 72
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

gen_learning_rate_val = args.lr
dis_learning_rate_val = args.lr / 10
edge_scale = args.s

SAVEITER = 1e4
DISPITER = 10
VALITER = 1000
VALREP = 2
MAXITER = 50e3

data_path = '../data/'
trainset_path = data_path + 'trainset.txt'
testset_path  = data_path + 'testset.txt'
dataset_path = '/scratch/data/Places365/data_large/a/'
# dataset_path = 'D:/deep learning/StreetView/'
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
d = data.dataset(batch_size, img_size, img_size, hiding_size, hiding_size, overlap_size, edge_scale)

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

# For contour generation
if args.pre==1:
    print("compute stage 2 loss.")

    l1_loss = tf.reduce_mean(tf.abs(x2_stage - imgs_pos))
    loss_recon = l1_loss
    
    # gan loss
    pos_set = imgs_pos
    gen_set = x2_stage
    adv_pos, adv_pos_dense = model.build_conditional_adversarial(pos_set, reuse=tf.AUTO_REUSE)
    adv_gen, adv_gen_dense = model.build_conditional_adversarial(gen_set, reuse=tf.AUTO_REUSE)
    loss_adv_D = tf.reduce_mean(tf.nn.l2_loss(adv_pos - tf.ones_like(adv_pos))) + tf.reduce_mean(tf.nn.l2_loss(adv_gen - tf.zeros_like(adv_gen)))
    loss_adv_G = tf.reduce_mean(tf.nn.l2_loss(adv_gen_dense - tf.ones_like(adv_gen_dense)))
    
    loss_D = loss_adv_D
    loss_G = (1 - gan_loss_alpha) * l1_loss + loss_adv_G * gan_loss_alpha

else:
    print("compute stage 1 loss.")
    l1_loss = tf.reduce_mean(tf.abs(x1_stage - imgs_pos))
    loss_G = l1_loss


if args.pre == 1:
    var_G = [v for v in tf.trainable_variables() if v.name.startswith('HFN')]
    var_D = [v for v in tf.trainable_variables() if v.name.startswith('DIS')]

else:
    var_G = [v for v in tf.trainable_variables() if v.name.startswith('LFN')]
    if args.t == 2:
        var_REP = [v for v in tf.trainable_variables() if v.name.startswith('REP')]
        var_G = var_REP + var_G

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())

# Definition of loss function
optimizer_G = tf.train.AdamOptimizer( learning_rate=d.gen_learning_rate, beta1=0.5, beta2=0.9)
grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
grads_vars_G = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G))
train_op_G = optimizer_G.apply_gradients( grads_vars_G )

if args.pre == 1:
    optimizer_D = tf.train.AdamOptimizer( learning_rate=d.dis_learning_rate, beta1=0.5, beta2=0.9)
    grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
    grads_vars_D = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D))
    train_op_D = optimizer_D.apply_gradients( grads_vars_D )

# Setup save/restore

tf.global_variables_initializer().run()

#TODO Load net seems not work
origiter = saver.iter
rs = np.random.RandomState(0)
loss_D_val = 0.
loss_G_val = 0.

train_batch_nums = len(trainset) // batch_size
test_batch_nums = len(testset) // batch_size
iters = origiter

if origiter > 0:
    util.loadNet(saver.latest,model,sess)
    if os.path.isfile('../wts/train/' + mission_path + 'popt.npz'):
        util.loadAdam('../wts/train/' + mission_path + 'popt.npz',opt,model.weights,sess)
    for k in range( (origiter + train_batch_nums - 1) // train_batch_nums):
        idx = rs.permutation(len(trainset))
    util.mprint("Restored to iteration %d" % origiter)    

# show display categories

show_train_nms = ['Loss_G']
show_test_nms = ['Loss_G_V']

if args.pre == 1:
    show_train_nms = ['Loss_Recon', 'Loss_G','Loss_D', 'Loss_adv_G']
    show_test_nms = ['Loss_Recon_V', 'Loss_G_V','Loss_D_V', 'Loss_adv_G_V']

print("trainset length: %d"%len(trainset))

while iters <= MAXITER:
    #TODO dont loop on epoch    
    if iters % train_batch_nums == 0:
        idx = rs.permutation(len(trainset))

    if iters == MAXITER:
        break

    if iters % 100 == 0:
        test_image_paths = testset[:batch_size]

        if args.pre==1:
            pre_out, output, loss_recon_val, loss_G_val, loss_D_val, loss_adv_G_val = sess.run(
                    [x1_stage, x2_stage, loss_recon, loss_G, loss_D, loss_adv_D],
                    feed_dict=d.dict(test_image_paths, gen_learning_rate_val, dis_learning_rate_val, False))
        else:
            contour_imgs, output, loss_G_val = sess.run(
                    [d.contour, x1_stage, loss_G],
                    feed_dict=d.dict(test_image_paths, gen_learning_rate_val, dis_learning_rate_val, False))
        # Generate result every 1000 iterations
        if iters % 2000 == 0:
            ii = 0
            # test_images = map(lambda x: util.load_gan_test_image(x), test_image_paths)
            output = list(output)
            idx_test = np.array([i for i in range(5)])
            for i in idx_test:
                # rec_con = rec_con + rec_hid
                if iters == 0:
                    if args.pre == 1:
                        pre_out_img = (255. * (1 + pre_out[i]) / 2.).astype(int)
                        skimage.io.imsave( os.path.join(result_path, 'img_pre_' + str(ii) +'.jpg'), pre_out_img)
                    else:
                        contour_img = (255. * contour_imgs[i]).astype(int)
                        skimage.io.imsave( os.path.join(result_path, 'img_contour_' + str(ii) +'.jpg'), contour_img[:, :, 0])
                output_img = (255 * (1 + output[i]) / 2).astype(int)
                skimage.io.imsave( os.path.join(result_path, 'img_'+str(ii)+ '_' + str(iters//1000) +'.jpg'), output_img)
                ii += 1

        
        show_test_vals = [loss_G_val]
        if args.pre == 1:
            show_test_vals = [loss_recon_val, loss_G_val, loss_D_val, loss_adv_G_val]
        util.vprint(iters, ['gen_lr', 'dis_lr'] + show_test_nms, [gen_learning_rate_val, dis_learning_rate_val] + show_test_vals)

        if np.isnan(loss_G_val.min() ) or np.isnan(loss_G_val.max()):
            print ("NaN detected!!")
            os._exit()
    
    image_paths = [trainset[idx[(iters % train_batch_nums) * batch_size + b]] for b in range(batch_size)]
    
    if args.pre == 1:
        for _ in range(2):
            _, loss_D_val = sess.run(
                [train_op_D, loss_D],
                feed_dict=d.dict(image_paths, gen_learning_rate_val, dis_learning_rate_val, True))

        _, loss_recon_val, loss_G_val, loss_adv_G_val = sess.run(
            [train_op_G, loss_recon, loss_G, loss_adv_G],
            feed_dict=d.dict(image_paths, gen_learning_rate_val, dis_learning_rate_val, True))
    else:
        _, loss_G_val = sess.run(
            [train_op_G, loss_G],
            feed_dict=d.dict(image_paths, gen_learning_rate_val, dis_learning_rate_val, True))


    # # Discriminator of GAN is updated only once in 100 iterations
    # if iters % 100  == 0:
    #     _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
    #             [train_op_D, loss_D, adversarial_pos, adversarial_neg],
    #             feed_dict=d.dict(image_paths, random_x, random_y, gen_learning_rate_val, dis_learning_rate_val, True))
            
    # display the output
    if iters % DISPITER == 0:
        show_train_vals = [loss_G_val]
        if args.pre == 1:
            show_train_vals = [loss_recon_val, loss_G_val, loss_D_val, loss_adv_G_val]
        # print ("Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max())
        util.vprint(iters, ['gen_lr', 'dis_lr'] + show_train_nms, [gen_learning_rate_val, dis_learning_rate_val] + show_train_vals)
        if util.stop:
            break

    # saving module
   
    iters += 1

    if iters % 1000 == 0:
        util.saveNet('../wts/train/' + mission_path + 'model_%d.npz'%iters,model,sess)
        saver.clean(every=SAVEITER,last=1)
        util.mprint('Saved Model')
        util.saveAdam('../wts/train/' + mission_path + 'popt_G.npz',optimizer_G, var_G, sess)
        util.mprint("Saved Optimizer.")
        if args.pre == 1:
            util.saveAdam('../wts/train/' + mission_path + 'popt_D.npz',optimizer_D, var_D, sess)
            util.mprint("Saved Optimizer.")
    
if iters > saver.iter:
    util.saveNet('../wts/train/' + mission_path + 'model_%d.npz'%iters,model,sess)
    saver.clean(every=SAVEITER,last=1)
    util.mprint('Saved Model')

# W_G here is a list of tensor
if iters > origiter:
    util.saveAdam('../wts/train/' + mission_path + 'popt_G.npz',optimizer_G, var_G, sess)
    util.mprint("Saved Optimizer.")
    if args.pre == 1:
        util.saveAdam('../wts/train/' + mission_path + 'popt_D.npz',optimizer_D, var_D, sess)
        util.mprint("Saved Optimizer.")

