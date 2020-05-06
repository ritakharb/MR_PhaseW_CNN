# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:30:08 2020

@author: ritak
"""

import os

#TRAINING PARAMETERS
data_folder = "data_folder" 
net_name = 'QSMnet+'
PS = '64' # patch_size
val_sub_num = 32 # number of validation datasets
val_input_path =  data_folder +  'Train/Input/val_input' 
val_label_path = data_folder +  'Train/Label/val_label' 
C = {
    'data': {
        'data_folder': data_folder,
        'train_data_path': data_folder + 'Train/Training_data_patch/training_data_patch_' + net_name + '_' + PS + '.hdf5',
        'save_path': data_folder + 'Checkpoints/' + net_name + '_'+ PS + '/'

    },

    'train': {
        'batch_size': 8, # batch size
        'learning_rate': 0.01, # learning rate limit
        'train_epochs': 15, # The number of training epochs
        'save_step': 5 # Step for saving network
    },
    'validation': {
        'display_step': 1, # display step of validation images
    }
}

train_data_path = C['data']['train_data_path']

save_path = C['data']['save_path']
if not os.path.exists(save_path+ 'validation_result'):
    os.makedirs(save_path + 'validation_result')

batch_size = C['train']['batch_size']
learning_rate = C['train']['learning_rate']
train_epochs = C['train']['train_epochs']
save_step = C['train']['save_step']

display_step = C['validation']['display_step']



#Utils
import tensorflow as tf
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io
import os
import nibabel as nib
import random
from tqdm import tqdm
if hasattr(tqdm,'_instances'):
    tqdm._instances.clear()

#
# Description:
#  Loss function for training QSMnet and QSMnet+
val_sub_num = 32
f2i = [None]*val_sub_num
f2l = [None]*val_sub_num
wrphase = [None]*val_sub_num
unwrphase = [None]*val_sub_num
cost_function_path = data_folder + 'learning_curve_data.txt'
#%% Dataset class
class dataset():
    def __init__(self):
        f = h5py.File(train_data_path, "r")
        f3 = scipy.io.loadmat(save_path + 'norm_factor.mat')

        self.trfield = f['temp_i']
        self.trsusc = f['temp_l']
        self.X_mean = f3["input_mean"]
        self.X_std = f3["input_std"]
        self.Y_mean = f3["label_mean"]
        self.Y_std = f3["label_std"]

for i in range(0, val_sub_num):
    f2i[i] = nib.load(val_input_path + str(i+1) + '.nii.gz')
    f2l[i] = nib.load(val_label_path + str(i+1) + '.nii.gz')
    wrphase[i] = f2i[i].get_fdata()
    unwrphase[i] = f2l[i].get_fdata()
    wrphase[i] = np.expand_dims(wrphase[i], axis=0)
    wrphase[i] = np.expand_dims(wrphase[i], axis=4)
    unwrphase[i] = np.expand_dims(unwrphase[i], axis=0)
    unwrphase[i] = np.expand_dims(unwrphase[i], axis=4)





#%% batch normalization, 3D convoluation, deconvolution, max pooling
def batch_norm(x, channel, isTrain, decay=0.99, name="bn"):

   with tf.compat.v1.variable_scope(name):
      beta = tf.compat.v1.get_variable(initializer=tf.constant(0.0, shape=[channel]), name='beta')
      gamma = tf.compat.v1.get_variable(initializer=tf.constant(1.0, shape=[channel]), name='gamma')
      batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
      mean_sh = tf.compat.v1.get_variable(initializer=tf.zeros([channel]), name="mean_sh", trainable=False)
      var_sh = tf.compat.v1.get_variable(initializer=tf.ones([channel]), name="var_sh", trainable=False)

      def mean_var_with_update():
         mean_assign_op = tf.compat.v1.assign(mean_sh, mean_sh * decay + (1 - decay) * batch_mean)
         var_assign_op = tf.compat.v1.assign(var_sh, var_sh * decay + (1 - decay) * batch_var)
         with tf.control_dependencies([mean_assign_op, var_assign_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(tf.cast(isTrain, tf.bool), mean_var_with_update, lambda: (mean_sh, var_sh))
      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name="normed")

   return normed



#%% Loss function
def l1(x, y):

    l1 = tf.reduce_mean(tf.reduce_mean(tf.abs(x - y), [1, 2, 3, 4]))

    return l1

def l2(x, y):

    l2 = tf.reduce_mean(tf.reduce_mean(tf.abs((x - y)**2), [1, 2, 3, 4]))

    return l2

#%%


input_val = [None]*val_sub_num
label_val = [None]*val_sub_num
cost_val = [None]*val_sub_num
im_val = [None]*val_sub_num
#total_train_cost_per_epoch = []
#total_val_cost_per_epoch =[]
#%% Training process
def Training_network(wrphase, unwrphase, dataset, X, Y, X_val, Y_val, predX_val, loss, loss_val, train_op, keep_prob, net_saver):
    with tf.compat.v1.Session() as sess:
        X_mean = dataset.X_mean
        X_std = dataset.X_std
        Y_mean = dataset.Y_mean
        Y_std = dataset.Y_std
        #%% Intializaion of all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        #%% Training
        print("Training Start!")
        ind = list(range(len(dataset.trfield)))

        file_lcurve = open(cost_function_path,"w+")
        file_lcurve.write("training cost\tvalidation cost\n")
        file_lcurve.close()
        for epoch in range(train_epochs):
            file_lcurve = open(cost_function_path,"a+")
            random.shuffle(ind)
            avg_cost = 0
            total_batch = int(len(ind)/batch_size)
            for i in tqdm(range(0, len(ind), batch_size)):
                ind_batch = ind[i:i + batch_size]
                ind_batch = np.sort(ind_batch)
                x_batch = (dataset.trfield[ind_batch, :, :, :, :] - X_mean) / X_std
                y_batch = (dataset.trsusc[ind_batch, :, :, :, :] - Y_mean) / Y_std
                cost, _ = sess.run([loss, train_op],
                                            feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
                avg_cost += cost / total_batch
            #total_train_cost_per_epoch[epoch+1]= avg_cost#training cost array
            #print("Epoch:", '%04d' % (epoch+1), "Training_cost=", "{:.5f}".format(avg_cost))
            file_lcurve.write("{:.5f}".format(avg_cost)+"\t")
            #%% Save network
            if (epoch + 1) % save_step == 0:
                net_saver.save(sess, save_path + net_name + '_' + str(PS), global_step = epoch + 1)

            #%% Validation

            #if (epoch + 1) % display_step == 0:
            total_val_cost_per_epoch = 0
            for i in range(0, val_sub_num):
                input_val[i] = (wrphase[i]  - X_mean ) / X_std
                label_val[i] = (unwrphase[i] - Y_mean ) / Y_std
                im_val[i], cost_val[i] = sess.run([predX_val, loss_val],
                                        feed_dict={X_val: input_val[i], Y_val: label_val[i], keep_prob : 1.0})
                total_val_cost_per_epoch += cost_val[i]/val_sub_num
                im_val[i] = dataset.Y_std * im_val[i].squeeze() + dataset.Y_mean
                scipy.io.savemat(save_path + 'validation_result/im_epoch' + str(epoch+1) + 'val' + str(i) + '.mat', mdict={'val_pred': im_val[i]})
            file_lcurve.write("{:.5f}".format(total_val_cost_per_epoch)+"\n")
            file_lcurve.close()


#%% Utils for inference

def save_nii(data, save_folder, name):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    nifti_affine = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]], dtype=np.float)

    #data = data.squeeze().numpy()
    data = np.fliplr(data)
    data = np.pad(data, ((2, 2), (6, 7), (6, 7)), mode='constant')
    nifti = nib.Nifti1Image(data, affine=nifti_affine)
    nib.save(nifti, os.path.join(save_folder, name + '.nii.gz'))

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    pad_field = np.expand_dims(pad_field, axis=0)
    pad_field = np.expand_dims(pad_field, axis=4)
    return pad_field, N_dif, N_16


def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final

def display_slice_inf(display_num, Pred):
     fig = plt.figure(figsize=(12,10))
     nonorm = matplotlib.colors.NoNorm()
     col = np.size(display_num)
     for i in range(col):

         subplot = fig.add_subplot(3, col, i + 1)
         subplot.set_xticks([]), subplot.set_yticks([])
 
         if i == 0:
             subplot.set_ylabel('Prediction', fontsize=18)

     plt.show()
     plt.close()


#%% Previous version
def conv3d(x, w_shape, b_shape, act_func, isTrain):
    weights = tf.compat.v1.get_variable("conv_weights", w_shape,
                              initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=isTrain)
    conv_3d = tf.nn.conv3d(x, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.compat.v1.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=isTrain)

    conv_3d = tf.nn.bias_add(conv_3d, biases)
    channel = conv_3d.get_shape().as_list()[-1]
    bn_x = batch_norm(conv_3d, channel, isTrain)

    if act_func is 'relu':
        return tf.nn.relu(bn_x)
    elif act_func is 'leaky_relu':
        return tf.nn.leaky_relu(bn_x, alpha = 0.1)

def conv(x, w_shape, b_shape, isTrain):
    weights = tf.compat.v1.get_variable("weights", w_shape,
                              initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=isTrain)
    biases = tf.compat.v1.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=isTrain)
    return tf.nn.conv3d(x, weights, strides=[1, 1, 1, 1, 1], padding='SAME') + biases

def max_pool(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')

def deconv3d(x, w_shape, b_shape, stride, isTrain):
    x_shape = tf.shape(x)
    weights = tf.compat.v1.get_variable("deconv_weights", w_shape,
                              initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=isTrain)
    biases = tf.compat.v1.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=isTrain)

    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, weights, output_shape, strides=[1, stride, stride, stride, 1],
                                  padding='SAME') + biases


#%%Network Model
#from utils import *
#
# Description:
#  3D U-net architecture of QSMnet+

def qsmnet_deep(x, act_func, reuse, isTrain):
    with tf.compat.v1.variable_scope("qsmnet", reuse=reuse) as scope:
        with tf.compat.v1.variable_scope("conv11", reuse=reuse) as scope:
            conv11 = conv3d(x, [5, 5, 5, 1, 32], [32], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv12", reuse=reuse) as scope:
            conv12 = conv3d(conv11, [5, 5, 5, 32, 32], [32], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool1", reuse=reuse) as scope:
            pool1 = max_pool(conv12, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv21", reuse=reuse) as scope:
            conv21 = conv3d(pool1, [5, 5, 5, 32, 64], [64], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv22", reuse=reuse) as scope:
            conv22 = conv3d(conv21, [5, 5, 5, 64, 64], [64], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool2", reuse=reuse) as scope:
            pool2 = max_pool(conv22, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv31", reuse=reuse) as scope:
            conv31 = conv3d(pool2, [5, 5, 5, 64, 128], [128], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv32", reuse=reuse) as scope:
            conv32 = conv3d(conv31, [5, 5, 5, 128, 128], [128], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool3", reuse=reuse) as scope:
            pool3 = max_pool(conv32, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv41", reuse=reuse) as scope:
            conv41 = conv3d(pool3, [5, 5, 5, 128, 256], [256], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv42", reuse=reuse) as scope:
            conv42 = conv3d(conv41, [5, 5, 5, 256, 256], [256], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool4", reuse=reuse) as scope:
            pool4 = max_pool(conv42, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("l_conv1", reuse=reuse) as scope:
            l_conv1 = conv3d(pool4, [5, 5, 5, 256, 512], [512], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("l_conv2", reuse=reuse) as scope:
            l_conv2 = conv3d(l_conv1, [5, 5, 5, 512, 512], [512], act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv4", reuse=reuse) as scope:
            deconv4 = deconv3d(l_conv2, [2, 2, 2, 256, 512], [256], 2, isTrain)
            deconv_concat4 = tf.concat([conv42, deconv4], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv51", reuse=reuse) as scope:
            conv51 = conv3d(deconv_concat4, [5, 5, 5, 512, 256], [256], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv52", reuse-reuse) as scope:
            conv52 = conv3d(conv51, [5, 5, 5, 256, 256], [256], act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv3", reuse=reuse) as scope:
            deconv3 = deconv3d(conv52, [2, 2, 2, 128, 256], [128], 2, isTrain)
            deconv_concat3 = tf.concat([conv32, deconv3], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv61", reuse=reuse) as scope:
            conv61 = conv3d(deconv_concat3, [5, 5, 5, 256, 128], [128], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv62", reuse=reuse) as scope:
            conv62 = conv3d(conv61, [5, 5, 5, 128, 128], [128], act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv2", reuse=reuse) as scope:
            deconv2 = deconv3d(conv62, [2, 2, 2, 64, 128], [64], 2, isTrain)
            deconv_concat2 = tf.concat([conv22, deconv2], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv71", reuse=reuse) as scope:
            conv71 = conv3d(deconv_concat2, [5, 5, 5, 128, 64], [64], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv72", reuse=reuse) as scope:
            conv72 = conv3d(conv71, [5, 5, 5, 64, 64], [64], act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv1", reuse=reuse) as scope:
            deconv1 = deconv3d(conv72, [2, 2, 2, 32, 64], [32], 2, isTrain)
            deconv_concat1 = tf.concat([conv12, deconv1], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv81", reuse=reuse) as scope:
            conv81 = conv3d(deconv_concat1, [5, 5, 5, 64, 32], [32], act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv82", reuse=reuse) as scope:
            conv82 = conv3d(conv81, [5, 5, 5, 32, 32], [32], act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("out", reuse=reuse) as scope:
            out_image = conv(conv82, [1, 1, 1, 32, 1], [1], isTrain)
            scope.reuse_variables()

    return out_image

#%%Training Data Patch

import numpy as np
import time
import os
#
# Description:
# Patching training data
#
# Inputs:
#   training_bf_patch_dir : directory of training data before patch
#                         : inputs - local field data (multiphs)
#                         : outputs - cosmos susceptibility data (multicos)
#   mask_dir : directory of mask data (mask)
#   PS : patch size
#   sub_num : subject number
#   aug_num : augmentation number
#   patch_num : the number of patches [x,y,z] per subject
#
# Outputs:
#   save training_data_patch_64.mat
#

'''
File Path
'''
FILE_PATH_INPUT = data_folder + 'Train/Input/train_input'#changed back
FILE_PATH_MASK = data_folder + 'Train/Input/mask1.mat'
FILE_PATH_LABEL = data_folder + 'Train/Label/train_label' #changed back
start_time = time.time()

'''
Constant Variables
'''
PS = 64  # Patch size
net_name = 'QSMnet+'
sub_num = 160 # number of subjects
dir_num = 1  # number of directions
patch_num = [5, 5, 5]  # Order of Dimensions: [x, y, z]

'''
Code Start
'''

# Create Result File
result_file = h5py.File(str(data_folder) + 'Train/Training_data_patch/training_data_patch_' + str(net_name) + '_' + str(PS) + '.hdf5', 'w')

# Patch the input & mask file ----------------------------------------------------------------


print("####patching input####")
patches = []
patches_mask = []
for dataset_num in range(1, sub_num + 1):
    field = nib.load(FILE_PATH_INPUT  + str (dataset_num) + '.nii.gz') #changed loading
    mask = scipy.io.loadmat(FILE_PATH_MASK)# + str(dataset_num) + '.nii.gz') #changed to loading
    matrix_size = field.shape
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    for idx in range(dir_num):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches.append(field.get_fdata()[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS])
                                   #idx])
                    patches_mask.append(mask['ones_mask_mat'][
                                        i * strides[0]:i * strides[0] + PS,
                                        j * strides[1]:j * strides[1] + PS,
                                        k * strides[2]:k * strides[2] + PS])
                                        #idx])
print("Done!")

patches = np.array(patches, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='float32', copy=False)

patches = np.expand_dims(patches,axis=4)
patches_mask = np.expand_dims(patches_mask,axis=4)
print("Final input data size : " + str(np.shape(patches)))

input_mean = np.mean(patches[patches_mask > 0])
input_std = np.std(patches[patches_mask > 0])

result_file.create_dataset('temp_i', data=patches)
result_file.create_dataset('temp_m', data=patches_mask)

del patches

# Patch the label file --------------------------------------------------------------------

patches = []
print("####patching label####")
for dataset_num in range(1, sub_num + 1):
    susc = nib.load(FILE_PATH_LABEL  + str (dataset_num) + '.nii.gz') #changed loading
    matrix_size = susc.shape
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    for idx in range(dir_num):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches.append(susc.get_fdata()[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS])
                                   #idx])
print("Done!")

patches = np.array(patches, dtype='float32', copy=False)
patches = np.expand_dims(patches,axis=4)
print("Final label data size : " + str(np.shape(patches)))

label_mean = np.mean(patches[patches_mask > 0])
label_std = np.std(patches[patches_mask > 0])
result_file.create_dataset('temp_l', data=patches)

del patches
del patches_mask
result_file.close()

save_path = data_folder + 'Checkpoints/' + str(net_name) + '_'+ str(PS) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
scipy.io.savemat(save_path + 'norm_factor.mat',
                 mdict={'input_mean': input_mean, 'input_std': input_std,
                        'label_mean': label_mean, 'label_std': label_std})
print("Total Time: {:.3f}".format(time.time() - start_time))


#%% Train

import time

#from training_params import *
#from utils import *
#from network_model import *

# Description:
#  Training code of QSMnet and QSMnet+


#%% Train
def train():
    tf.compat.v1.reset_default_graph()
    #%% Loading dataset
    train_dataset = dataset() # Training set, validation set

    #%% Declaration of tensor, [None, PS, PS, PS] - PS is the size in x,y,z direction
    X = tf.compat.v1.placeholder("float", [None, PS, PS, PS, 1]) # Training input
    Y = tf.compat.v1.placeholder("float", [None, PS, PS, PS, 1]) # Training label

    N = np.shape(wrphase[1]) # matrix size of validation set
    X_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1]) # Validation input
    Y_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1]) # Validation label
    keep_prob = tf.compat.v1.placeholder("float") #dropout rate

    #%% Definition of model
    predX = qsmnet_deep(X, 'relu', False, True)
    predX_val = qsmnet_deep(X_val, 'relu', True, False)

    #%% Definition of loss function
    loss = l1(predX, Y)
    loss_val = l1(predX_val,Y_val)
    #loss = l2(predX, Y)
    #loss_val = l2(predX_val,Y_val)

    #%% Definition of optimizer
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #%% Generate saver instance
    qsm_saver = tf.compat.v1.train.Saver()

    #%% Running session
    Training_network(wrphase, unwrphase, train_dataset, X, Y, X_val, Y_val, predX_val, loss, loss_val, train_op, keep_prob, qsm_saver)

if __name__ == '__main__':
    start_time = time.time()
    train()
    print("Total training time : {} sec".format(time.time() - start_time))




#%%Inference


#
# Description :
#   Inference code of QSMnet and QSMnet+
#   Save susceptibility map in Matlab and NII format
# Outputs :
#   results_<network_name>.mat & results_<network_name>.nii
#   ppm unit
'''
Network model
'''
network_name = 'QSMnet+_64'
net_model = 'qsmnet_deep'
sub_num = 12 #number of subjects in testset

'''
File Path
'''
FILE_PATH_INPUT = data_folder + 'Test/Input/test_input'
FILE_PATH_PRED = data_folder + 'Test/Prediction/'

def inf():
    f = scipy.io.loadmat(data_folder + 'Checkpoints/'+ network_name + '/' + 'norm_factor.mat')


    b_mean = f['input_mean']
    b_std = f['input_std']
    y_mean = f['label_mean']
    y_std = f['label_std']

    for i in range(1, sub_num + 1):
        input_data = nib.load(FILE_PATH_INPUT + str(i) +'.nii.gz')
        tf.compat.v1.reset_default_graph()

        print('Subject number: ' + str(i))
        field = input_data.get_fdata()
        field = (field - b_mean) / b_std
        [pfield, N_difference, N] = padding_data(field)

        Z = tf.compat.v1.placeholder("float", [None, N[0], N[1], N[2], 1])
        keep_prob = tf.compat.v1.placeholder("float")

        net_func = qsmnet_deep
        feed_result = net_func(Z, 'relu', False, False)

        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            print('##########Restore Network##########')
            saver.restore(sess, data_folder + 'Checkpoints/'+ network_name + '/' + network_name +  '-'+ str(train_epochs))
            print('Done!')
            print('##########Inference...##########')
            result_im = y_std * sess.run(feed_result, feed_dict={Z: pfield, keep_prob: 1.0}) + y_mean
            result_im = crop_data(result_im.squeeze(), N_difference)

            print('##########Saving MATLAB & NII file...##########')
            scipy.io.savemat(FILE_PATH_PRED + '/subject' + str(i) + '_' + str(network_name) + '-'+ str(train_epochs)+'_relu'+'.mat', mdict={'sus': result_im})
            #save_nii(result_im, FILE_PATH_PRED, 'subject' + str(i) + '_' + str(network_name)+'-'+ str(train_epochs)) # incorrect way of saving NII, introduces dimension discrepency 
            nib.save(nib.Nifti1Image(result_im, np.eye(4)),FILE_PATH_PRED + 'subject' + str(i) + '_' + str(network_name)+'-'+ str(train_epochs) + '_relu')
        print('All done!')

if __name__ == '__main__':
  start_time = time.time()
  inf()
  print("Total inference time : {} sec".format(time.time() - start_time))
