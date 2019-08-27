# from skimage import io
from PIL import ImageFile
import numpy as np
import re
import os
import time
import sys
import signal
from glob import glob
import numpy as np
import tensorflow as tf
from IPython import embed
import math
import scipy.stats as st
from skimage import feature

# This part of codes come from Ayan Chakrabarti <ayan@wustl.edu>
# General usage

# Logic for trapping ctrlc and setting stop
stop = False
_orig = None
def handler(a,b):
    global stop
    stop = True
    signal.signal(signal.SIGINT,_orig)
_orig = signal.signal(signal.SIGINT,handler)


def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()

def vprint(it, nms, vals):
    s = '[%06d]' % it
    for i in range(len(nms)):
        s = s + ' ' + nms[i] + ' = %.3e'%vals[i]
    mprint(s)
    
# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])

# Save/load networks
def saveNet(fn,net,sess):
    wts = {}
    for k in net.weights.keys():
        wts[k] = net.weights[k].eval(sess)
    np.savez(fn,**wts)

def loadNet(fn,net,sess):
    wts = np.load(fn)
    for k in wts.keys():
        if k in net.weights:
            print("loading Net")
            wvar = net.weights[k]
            wk = wts[k].reshape(wvar.get_shape())
            wvar.load(wk,sess)
    print("Loading finished!")


# Save/load Adam optimizer state
def saveAdam(fn,opt,vdict,sess):
    weights = {}
    beta1_power, beta2_power = opt._get_beta_accumulators()
    weights['b1p'] = beta1_power.eval(sess)
    weights['b2p'] = beta2_power.eval(sess)
    for v in vdict:
        weights['m_%s' % v.name] = opt.get_slot(v,'m').eval(sess)
        weights['v_%s' % v.name] = opt.get_slot(v,'v').eval(sess)
    np.savez(fn,**weights)


def loadAdam(fn,opt,vdict,sess):
    weights = np.load(fn)
    beta1_power, beta2_power = opt._get_beta_accumulators()
    beta1_power.load(weights['b1p'],sess)
    beta2_power.load(weights['b2p'],sess)

    for v in vdict:
        opt.get_slot(v,'m').load(weights['m_%s' % v.name],sess)
        opt.get_slot(v,'v').load(weights['v_%s' % v.name],sess)


#Content Encoder script
################ This part has been rewritten in data.py. This comment part is used for debug and review

# def load_gan_test_image( path, pre_height=267, pre_width=267, height=256, width=256 ):
#     try:
#         img = io.imread( path ).astype( float )
#     except:
#         return None

#     img /= 255.

#     if img is None: return None
#     if len(img.shape) < 2: return None
#     if len(img.shape) == 4: return None
#     if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
#     if img.shape[2] == 4: img=img[:,:,:3]
#     if img.shape[2] > 4: return None
    

#     short_edge = min( img.shape[:2] )
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
#     resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

#     rand_y = np.random.randint(0, pre_height - height)
#     rand_x = np.random.randint(0, pre_width - width)

#     resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

#     return (resized_img * 2)-1 #(resized_img - 127.5)/127.5


def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs
    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def resize_mask_like(mask, x):
    """Resize mask like shape of x.
    Args:
        mask: Original mask.
        x: To shape of x.
    Returns:
        tf.Tensor: resized mask
    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):

    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    # print(x)
    return x

# def spatial_discounting_mask(gamma, c_height, c_width):
    
#     shape = [1, c_height, c_width, 1]    
#     # logger.info('Use spatial discounting l1 loss.')
#     mask_values = np.ones((c_height, c_width))
#     for i in range(c_height):
#         for j in range(c_width):
#             mask_values[i, j] = max(
#                 gamma**min(i, c_height-i),
#                 gamma**min(j, c_width-j))
#     mask_values = np.expand_dims(mask_values, 0)
#     mask_values = np.expand_dims(mask_values, 3)

#     return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((size, size, 1, 1))
    # out_filter = np.repeat(out_filter, [1, 1, inchannels, 1])
    return out_filter

def tf_make_guass_var(size, attention_gamma, inchannels=1, outchannels=1):
    kernel = gauss_kernel(size, attention_gamma, inchannels, outchannels)
    var = tf.Variable(tf.convert_to_tensor(kernel))
    return var

def spatial_discounting_mask(mask, attention_gamma=1.0/40, hsize=64,  iters=9):
    eps = 1e-5
    kernel = tf_make_guass_var(hsize, attention_gamma)
    init = 1-mask
    mask_priority = None
    mask_priority_pre = None
    for i in range(iters):
        mask_priority = tf.nn.conv2d(init, kernel, strides=[1,1,1,1], padding='SAME')
        mask_priority = mask_priority * mask
        if i == iters-2:
            mask_priority_pre = mask_priority
        init = mask_priority + (1-mask)
    mask_priority = mask_priority_pre / (mask_priority+eps)
    return mask_priority



################  Loss function zoo  ####################
def gan_ls_loss(pos, neg, value=1., name='gan_ls_loss'):
    """
    gan with least-square loss
    """
    with tf.variable_scope(name):
        l2_pos = tf.reduce_mean(tf.squared_difference(pos, value))
        l2_neg = tf.reduce_mean(tf.square(neg))
        # scalar_summary('pos_l2_avg', l2_pos)
        # scalar_summary('neg_l2_avg', l2_neg)
        d_loss = tf.add(.5 * l2_pos, .5 * l2_neg)
        g_loss = tf.reduce_mean(tf.squared_difference(neg, value))
        # scalar_summary('d_loss', d_loss)
        # scalar_summary('g_loss', g_loss)
    return g_loss, d_loss


def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.
    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        # scalar_summary('d_loss', d_loss)
        # scalar_summary('g_loss', g_loss)
        # scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        # scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss

def max_downsampling(x, ratio=2):
    iters = math.log2(ratio)
    assert int(iters) == iters
    for _ in range(int(iters)):
        x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME')
    return x

def mask_fill(mask, k = 16):
    gridy = [i for i in range(0, mask.shape[1], k)]
    gridx = [i for i in range(0, mask.shape[2], k)]
    grid = np.meshgrid(gridy, gridx)
    tmp_mask = np.zeros((mask.shape[1:3]))
    tmp_mask[tuple(grid)] = 1.0
    tmp_mask = np.expand_dims(tmp_mask, axis=2)
    tmp_mask = [tmp_mask for i in range(mask.shape[0])]
    tmp_mask = np.array(tmp_mask)
    sparse_points = tf.constant(tmp_mask, dtype=tf.float32)
    mask = mask + sparse_points
    return mask, sparse_points

def edge_detector(img_tensor, minRate=0.2, maxRate=0.2, 
             preserve_size=True, remove_high_val=False, return_raw_edges=False):
    
    MAX = 1    
    kernel_size = 3
    sigma  = 1.2
    def Gaussian_Filter(kernel_size=kernel_size, sigma=sigma): 
        k = (kernel_size-1)//2 
        x, y = np.meshgrid(np.linspace(-k, k, kernel_size), np.linspace(-k, k, kernel_size))
        g = np.exp(-(x * x + y * y) / (2 * sigma**2)) / (2 * np.pi * sigma ** 2)
        return np.asarray(g).reshape(kernel_size,kernel_size,1,1)

    gaussian_filter = tf.constant(Gaussian_Filter(kernel_size, sigma), tf.float32)                 #STEP-1
    h_filter = tf.reshape(tf.constant([[-1,0,1],[-2,0,2],[-1,0,1]], tf.float32), [3,3,1,1])    #STEP-2
    v_filter = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], tf.float32), [3,3,1,1])    #STEP-2

    np_filter_0 = np.zeros((3,3,1,2))
    np_filter_0[1,0,0,0], np_filter_0[1,2,0,1] = 1,1 ### Left & Right
    # print(np_filter_0)
    filter_0 = tf.constant(np_filter_0, tf.float32)
    np_filter_90 = np.zeros((3,3,1,2))
    np_filter_90[0,1,0,0], np_filter_90[2,1,0,1] = 1,1 ### Top & Bottom
    filter_90 = tf.constant(np_filter_90, tf.float32)
    np_filter_45 = np.zeros((3,3,1,2))
    np_filter_45[0,2,0,0], np_filter_45[2,0,0,1] = 1,1 ### Top-Right & Bottom-Left
    filter_45 = tf.constant(np_filter_45, tf.float32)
    np_filter_135 = np.zeros((3,3,1,2))
    np_filter_135[0,0,0,0], np_filter_135[2,2,0,1] = 1,1 ### Top-Left & Bottom-Right
    filter_135 = tf.constant(np_filter_135, tf.float32)
        
    np_filter_sure = np.ones([3,3,1,1]); np_filter_sure[1,1,0,0] = 0
    filter_sure = tf.constant(np_filter_sure, tf.float32)
    border_paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

    def Border_Padding(x, pad_width):
        for _ in range(pad_width): x = tf.pad(x, border_paddings, 'SYMMETRIC')
        return x

    def FourAngles(d):
        d0   = tf.to_float(tf.greater_equal(d,157.5))+tf.to_float(tf.less(d,22.5))
        d45  = tf.to_float(tf.greater_equal(d,22.5))*tf.to_float(tf.less(d,67.5))
        d90  = tf.to_float(tf.greater_equal(d,67.5))*tf.to_float(tf.less(d,112.5))
        d135 = tf.to_float(tf.greater_equal(d,112.5))*tf.to_float(tf.less(d,157.5))

        return (d0,d45,d90,d135)

    img_tensor = (img_tensor/tf.reduce_max(img_tensor))*MAX
    if preserve_size: img_tensor = Border_Padding(img_tensor, (kernel_size-1)//2)

    x_gaussian = tf.nn.convolution(img_tensor, gaussian_filter, padding='VALID')
    if remove_high_val: x_gaussian = tf.clip_by_value(x_gaussian, 0, MAX/2)
    
    if preserve_size: x_gaussian = Border_Padding(x_gaussian, 1)
    Gx = tf.nn.convolution(x_gaussian, h_filter, padding='VALID')
    Gy = tf.nn.convolution(x_gaussian, v_filter, padding='VALID')
    G = tf.sqrt(tf.square(Gx) + tf.square(Gy))
    BIG_PHI = tf.atan2(Gy,Gx)
    BIG_PHI    = (BIG_PHI*180/np.pi)%180         ### Convert from Radian to Degree
    D_0,D_45,D_90,D_135 = FourAngles(BIG_PHI)    ### Round the directions to 0, 45, 90, 135 (only take the masks)
    
    targetPixels_0 = tf.nn.convolution(G, filter_0, padding='SAME')
    isGreater_0 = tf.to_float(tf.greater(G*D_0, targetPixels_0))
    isMax_0 = isGreater_0[:,:,:,0:1]*isGreater_0[:,:,:,1:2]
    
    targetPixels_90 = tf.nn.convolution(G, filter_90, padding='SAME')
    isGreater_90 = tf.to_float(tf.greater(G*D_90, targetPixels_90))
    isMax_90 = isGreater_90[:,:,:,0:1]*isGreater_90[:,:,:,1:2]
    
    targetPixels_45 = tf.nn.convolution(G, filter_45, padding='SAME')
    isGreater_45 = tf.to_float(tf.greater(G*D_45, targetPixels_45))
    isMax_45 = isGreater_45[:,:,:,0:1]*isGreater_45[:,:,:,1:2]
    
    targetPixels_135 = tf.nn.convolution(G, filter_135, padding='SAME')
    isGreater_135 = tf.to_float(tf.greater(G*D_135, targetPixels_135))
    isMax_135 = isGreater_135[:,:,:,0:1]*isGreater_135[:,:,:,1:2]
    
    edges_raw = G*(isMax_0 + isMax_90 + isMax_45 + isMax_135)
    edges_raw = tf.clip_by_value(edges_raw, 0, MAX)
    
    edges_sure = tf.to_float(tf.greater_equal(edges_raw, maxRate))
    edges_weak = tf.to_float(tf.less(edges_raw, maxRate))*tf.to_float(tf.greater_equal(edges_raw, minRate))
    
    edges_connected = tf.nn.convolution(edges_sure, filter_sure, padding='SAME')*edges_weak
    for _ in range(10): edges_connected = tf.nn.convolution(edges_connected, filter_sure, padding='SAME')*edges_weak
    
    edges_final = edges_sure + tf.clip_by_value(edges_connected,0,MAX)
    mask = tf.to_float(tf.greater(edges_final, 0.2))
    # edges_final = edges_final * mask
    mask = tf.where(tf.greater(mask*255, 200), tf.ones_like(mask)*255, tf.zeros_like(mask))
    mask = mask / 255
    # embed()
    return mask