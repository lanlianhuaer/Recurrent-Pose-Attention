
####################
import sys

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.expr.normalize import CrossChannelNormalization
from lstm_lib import dropout_layer,lstm_param_init,joint_attention_lstm
from theano.tensor.nnet.bn import batch_normalization

import warnings
import os
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

class Weight(object):

    def __init__(self, w_shape, mean=0, std=0.01):
        super(Weight, self).__init__()
        if std != 0:
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)

    def save_weight(self, dir, name):
        # print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        if os.path.exists(dir + name + '.npy'):

            self.np_values = np.load(dir + name + '.npy')
            # test = self.np_values
            # test= np.load(dir + name + '.npy')
            # print 'test.shape',test.shape,'vs',self.np_values.shape
            # print('shape',self.val.get_value().shape)
            if self.np_values.shape==self.val.get_value().shape:
                self.val.set_value(self.np_values)
                print 'weight loaded: ' + name
            else:
                print('dimetion mismatch',self.np_values.shape,'to',self.val.get_value().shape,'failed')
        else:
            print 'warning: weight'+ name + ' not found,ignored... '
class DataLayer(object):

    def __init__(self, input, image_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the
        whole batch.
        '''

        # trick for random mirroring
        mirror = input[:, :, ::-1, :]
        input = T.concatenate([input, mirror], axis=0)

        # crop images
        center_margin = (image_shape[2] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
        self.output = self.output[
            :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

        print "data layer with shape_in: " + str(image_shape)
class ConvPoolLayer(object):

    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 group, poolsize, poolstride, bias_init, lrn=False,
                 lib_conv='cudnn',poolpadsize=(0,0),caffe_style=False,Bn=False,
                 ):
        '''
        lib_conv can be cudnn (recommended)or cudaconvnet
        '''

        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[0]
        self.lrn = lrn
        self.lib_conv = lib_conv
        # assert input.shape==image_shape
        assert group in [1, 2]

        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        if self.lrn:
            self.lrn_func = CrossChannelNormalization(alpha=0.0005, k=1)
            # self.lrn_func = CrossChannelNormalization(alpha=0.0005)

        if group == 1:
            self.W = Weight(self.filter_shape)
            self.b = Weight(self.filter_shape[3], bias_init, std=0)
        else:
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[3] = self.filter_shape[3] / 2
            self.image_shape[0] = self.image_shape[0] / 2
            self.image_shape[3] = self.image_shape[3] / 2
            self.W0 = Weight(self.filter_shape)
            self.W1 = Weight(self.filter_shape)
            self.b0 = Weight(self.filter_shape[3], bias_init, std=0)
            self.b1 = Weight(self.filter_shape[3], bias_init, std=0)

        if lib_conv == 'cudaconvnet':
            self.conv_op = FilterActs(pad=self.padsize, stride=self.convstride,
                                      partial_sum=1)
            # Conv
            if group == 1:
                contiguous_input = gpu_contiguous(input)
                contiguous_filters = gpu_contiguous(self.W.val)
                conv_out = self.conv_op(contiguous_input, contiguous_filters)
                conv_out = conv_out + self.b.val.dimshuffle(0, 'x', 'x', 'x')
            else:
                contiguous_input0 = gpu_contiguous(
                    input[:self.channel / 2, :, :, :])
                contiguous_filters0 = gpu_contiguous(self.W0.val)
                conv_out0 = self.conv_op(contiguous_input0, contiguous_filters0)
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle(0, 'x', 'x', 'x')

                contiguous_input1 = gpu_contiguous(
                    input[self.channel / 2:, :, :, :])
                contiguous_filters1 = gpu_contiguous(self.W1.val)
                conv_out1 = self.conv_op(
                    contiguous_input1, contiguous_filters1)
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle(0, 'x', 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=0)
            # ReLu
            self.output = T.maximum(conv_out, 0)
            # Pooling
            if self.poolsize != 1:
                self.pool_op = MaxPool(ds=poolsize, stride=poolstride)
                self.output = self.pool_op(self.output)

        elif lib_conv == 'cudnn':

            input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            if group == 1:
                W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )
                conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
            else:
                W0_shuffled = \
                    self.W0.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out0 = \
                    dnn.dnn_conv(img=input_shuffled[:, :self.channel / 2,
                                     :, :],
                                 kerns=W0_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out0 = conv_out0 + \
                            self.b0.val.dimshuffle('x', 0, 'x', 'x')
                W1_shuffled = \
                    self.W1.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out1 = \
                    dnn.dnn_conv(img=input_shuffled[:, self.channel / 2:,
                                     :, :],
                                 kerns=W1_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out1 = conv_out1 + \
                            self.b1.val.dimshuffle('x', 0, 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

            self.conv_out=conv_out
            if Bn:
                #Warning this just used for testing phase!!!!
                self.mean = theano.shared(value = np.zeros((1, filter_shape[3], 1,1), dtype=theano.config.floatX),broadcastable=[True,False,True,True], name='mean',
                                           borrow=True)
                self.var = theano.shared(value = np.ones((1, filter_shape[3], 1,1), dtype=theano.config.floatX),broadcastable=[True,False,True,True], name='var',
                                           borrow=True)

                self.gamma = theano.shared(value = np.ones((filter_shape[3],), dtype=theano.config.floatX), name='gamma',
                                           borrow=True)
                self.beta = theano.shared(value = np.zeros((  filter_shape[3], ), dtype=theano.config.floatX),name='beta',
                                          borrow=True)
                conv_out = batch_normalization(inputs = conv_out, gamma = self.gamma, beta = self.beta,
                                               mean = self.mean,
                                               std = T.sqrt(self.var),mode='high_mem')
            # ReLu
                self.Bn = conv_out
            self.output = T.maximum(conv_out, 0)
            # # Pooling
            if caffe_style:
                self.output=self.output[:,:,::-1,::-1]
            if self.poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize, poolsize),
                                           stride=(poolstride, poolstride),
                                           pad=poolpadsize)
            if caffe_style:
                self.output =self.output[:,:,::-1,::-1]

            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        else:
            NotImplementedError("lib_conv can only be cudaconvnet or cudnn")

        if group == 1:
            if Bn:
                #self.params = [self.W.val, self.b.val,self.beta,self.gamma,self.mean,self.var]
                self.params = [self.W.val, self.b.val]
                self.weight_type = ['W', 'b']
                #self.weight_type = ['W', 'b','b','b','b','b']
                pass
            else:
                self.params = [self.W.val, self.b.val]
                self.weight_type = ['W', 'b']
        else:
            self.params = [self.W0.val, self.b0.val, self.W1.val, self.b1.val]
            self.weight_type = ['W', 'b', 'W', 'b']

        print "conv ({}) layer with shape_in: {}".format(lib_conv,
                                                         str(image_shape))


class ConvPoolLayer_org(object):

    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 group, poolsize, poolstride, bias_init, lrn=False,
                 lib_conv='cudnn',poolpadsize=(0,0),caffe_style=False
                 ):
        '''
        lib_conv can be cudnn (recommended)or cudaconvnet
        '''

        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[0]
        self.lrn = lrn
        self.lib_conv = lib_conv
        assert group in [1, 2]

        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        if self.lrn:
            self.lrn_func = CrossChannelNormalization(alpha=0.0005, k=1, caffe_style=True)
            # self.lrn_func = CrossChannelNormalization(alpha=0.0005)

        if group == 1:
            self.W = Weight(self.filter_shape)
            self.b = Weight(self.filter_shape[3], bias_init, std=0)
        else:
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[3] = self.filter_shape[3] / 2
            self.image_shape[0] = self.image_shape[0] / 2
            self.image_shape[3] = self.image_shape[3] / 2
            self.W0 = Weight(self.filter_shape)
            self.W1 = Weight(self.filter_shape)
            self.b0 = Weight(self.filter_shape[3], bias_init, std=0)
            self.b1 = Weight(self.filter_shape[3], bias_init, std=0)

        if lib_conv == 'cudaconvnet':
            self.conv_op = FilterActs(pad=self.padsize, stride=self.convstride,
                                      partial_sum=1)

            # Conv
            if group == 1:
                contiguous_input = gpu_contiguous(input)
                contiguous_filters = gpu_contiguous(self.W.val)
                conv_out = self.conv_op(contiguous_input, contiguous_filters)
                conv_out = conv_out + self.b.val.dimshuffle(0, 'x', 'x', 'x')
            else:
                contiguous_input0 = gpu_contiguous(
                    input[:self.channel / 2, :, :, :])
                contiguous_filters0 = gpu_contiguous(self.W0.val)
                conv_out0 = self.conv_op(contiguous_input0, contiguous_filters0)
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle(0, 'x', 'x', 'x')

                contiguous_input1 = gpu_contiguous(
                    input[self.channel / 2:, :, :, :])
                contiguous_filters1 = gpu_contiguous(self.W1.val)
                conv_out1 = self.conv_op(
                    contiguous_input1, contiguous_filters1)
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle(0, 'x', 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=0)

            # ReLu
            self.output = T.maximum(conv_out, 0)
            # conv_out = gpu_contiguous(conv_out)

            # self.conv_out =self.output
            # self.input_shuffled = self.output
            # self.input_W =self.output
            # self.input_b = self.output


            # Pooling
            if self.poolsize != 1:
                self.pool_op = MaxPool(ds=poolsize, stride=poolstride)
                self.output = self.pool_op(self.output)

        elif lib_conv == 'cudnn':

            input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
            if group == 1:
                W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )
                conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
            else:
                W0_shuffled = \
                    self.W0.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out0 = \
                    dnn.dnn_conv(img=input_shuffled[:, :self.channel / 2,
                                                    :, :],
                                 kerns=W0_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out0 = conv_out0 + \
                    self.b0.val.dimshuffle('x', 0, 'x', 'x')
                W1_shuffled = \
                    self.W1.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
                conv_out1 = \
                    dnn.dnn_conv(img=input_shuffled[:, self.channel / 2:,
                                                    :, :],
                                 kerns=W1_shuffled,
                                 subsample=(convstride, convstride),
                                 border_mode=padsize,
                                 )
                conv_out1 = conv_out1 + \
                    self.b1.val.dimshuffle('x', 0, 'x', 'x')
                conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

            # ReLu
            self.output = T.maximum(conv_out, 0)
            # self.conv_out =self.output
            # self.input_shuffled = input_shuffled
            # self.input_W =W_shuffled
            # self.input_b = self.b.val.dimshuffle('x', 0, 'x', 'x')
             # LRN
            # if self.lrn:
            #     lrn_input = gpu_contiguous(self.output)
            #     self.output = self.lrn_func(self.output)
            # self.lrn_out =self.output
            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            if self.lrn:
            # lrn_input = gpu_contiguous(self.output)
                self.output = self.lrn_func(self.output)
                self.lrn_out =self.output
            self.output = self.output.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # Pooling
            if caffe_style:
                self.output=self.output[:,:,::-1,::-1]
            if self.poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize, poolsize),
                                           stride=(poolstride, poolstride),
                                           pad=poolpadsize)
            if caffe_style:
                self.output =self.output[:,:,::-1,::-1]

            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            # self.pool_out=self.output
        else:
            NotImplementedError("lib_conv can only be cudaconvnet or cudnn")

        # if self.lrn:
        #     lrn_input = gpu_contiguous(self.output)
            # self.output = self.lrn_func(self.output)
            # self.lrn_out =self.output

        if group == 1:
            self.params = [self.W.val, self.b.val]
            self.weight_type = ['W_lr', 'b_lr']
        else:
            self.params = [self.W0.val, self.b0.val, self.W1.val, self.b1.val]
            self.weight_type = ['W', 'b', 'W', 'b']

        print "conv ({}) layer with shape_in: {}".format(lib_conv,
                                                         str(image_shape))


class PoolLayer(object):

    def __init__(self, input,  poolsize, poolstride,lib_conv,poolpad=0,poolmode='max',caffe_style=False
                 ):
        '''
        lib_conv can be cudnn (recommended)or cudaconvnet
        '''
        self.poolsize=poolsize
        if lib_conv == 'cudaconvnet':
            # Pooling
            if self.poolsize != 1:
                self.pool_op = MaxPool(ds=poolsize, stride=poolstride)
                self.output = self.pool_op(input)

        elif lib_conv == 'cudnn':

            input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
            if caffe_style:
                self.output =input_shuffled[:,:,::-1,::-1]
            else:
                self.output =input_shuffled
                # poolpad=1
            if self.poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize, poolsize),
                                           stride=(poolstride, poolstride),
                                           mode=poolmode,pad=(poolpad,poolpad)
                                           )
            if caffe_style:
                self.output =self.output[:,:,::-1,::-1]

            self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            # self.pool_out=self.output
        else:
            NotImplementedError("lib_conv can only be cudaconvnet or cudnn")

        # if self.lrn:
        #     lrn_input = gpu_contiguous(self.output)
            # self.output = self.lrn_func(self.output)
            # self.lrn_out =self.output
class FCLayer(object):

    def __init__(self, input, n_in, n_out,relu=True):

        self.W = Weight((n_in, n_out), std=0.005)
        self.b = Weight(n_out, mean=0.1, std=0)
        self.input = input
        lin_output = T.dot(self.input, self.W.val) + self.b.val
        if relu:
            self.output = T.maximum(lin_output, 0)
        else:
            self.output =lin_output
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W_lr', 'b_lr']
        print 'fc layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out)

class JointAttentionLstmLayer(object):

    def __init__(self,config,num_joints,conv_fea, mask, batch_size, num_seq,trng, use_noise, n_in, n_out,dim_part, phase_test=False):
        n_timesteps = batch_size/num_seq
        lstm_options = {}
        lstm_options['reg_scale_x'] = config['reg_scale_x']
        lstm_options['reg_scale_y'] = config['reg_scale_y']
        lstm_options['n_timesteps']=n_timesteps
        lstm_options['num_seq']=num_seq
        lstm_options['dim_proj'] = n_out
        lstm_options['dim_part'] = dim_part
        if phase_test==True:
            lstm_options['use_dropout'] = False
        else:
            lstm_options['use_dropout'] = True

        lstm_options['encoder'] = 'lstm'
        lstm_options['dim_flow'] = n_in
        lstm_options['num_joints'] = num_joints
        lstm_options['conv_dim'] = 1024
        # batch_size*num_joints

        (self.LSTM_H_lamada_spa,self.LSTM_U_lamada_spa,
         self.LSTM_W_lamada_spa,self.LSTM_b_lamada_spa,
         self.LSTM_U_spa, self.LSTM_W_spa,
         self.LSTM_b_spa) = lstm_param_init(lstm_options)
        mask = mask.reshape([num_seq,n_timesteps])
        mask = mask.dimshuffle(1,0)
        print 'embing succeed...'
        proj,attention = joint_attention_lstm(conv_fea,
                                                  self.LSTM_H_lamada_spa,self.LSTM_U_lamada_spa,self.LSTM_W_lamada_spa,self.LSTM_b_lamada_spa,
                                                  self.LSTM_U_spa,self.LSTM_W_spa,self.LSTM_b_spa,
                                                  lstm_options,
                                                  prefix=lstm_options['encoder'],
                                                  mask=mask)
        proj = proj * mask[:, :, None]
        if lstm_options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)
        proj =proj.dimshuffle(1,0,2)
        attention=attention.dimshuffle(2,0,1,3)
        # 81* batchsize*13*51

        self.attention =attention.reshape([num_seq*n_timesteps,lstm_options['reg_scale_x']*lstm_options['reg_scale_y']+1,num_joints])
        proj = proj.reshape([n_timesteps*num_seq,int(lstm_options['dim_proj'])])
        self.output = proj
        self.params = [self.LSTM_H_lamada_spa,self.LSTM_U_lamada_spa,self.LSTM_W_lamada_spa,self.LSTM_b_lamada_spa,
                       self.LSTM_W_spa,self.LSTM_U_spa, self.LSTM_b_spa]
        self.weight_type = [
            'W','W', 'W', 'b',
            'W','W', 'b']
        print 'lstm layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out)
class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * self.prob_keep * input

        DropoutLayer.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)

class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'softmax layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)
    def negative_log_likelihood(self, y, mask):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]*mask)
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            # return T.mean(T.neq(self.y_pred, y))
            return T.mean(T.neq(self.y_pred, y), dtype=config.floatX)
        else:
            raise NotImplementedError()
    def errors_video(self, y, mask, batch_size,num_seq):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            # return T.mean(T.neq(self.y_pred, y))
            n_sample = num_seq
            # mask1=mask.reshape([batch_size/n_sample,n_sample])
            # mask1=mask.reshape([n_sample,batch_size/n_sample])
            # mask1 = mask.reshape([batch_size/n_sample,n_sample])
            # mask1 = mask1.dimshuffle(1,0)
            mask = mask.reshape([n_sample,batch_size/n_sample])
            length=theano.tensor.cast(mask.sum(axis=1),'int32')-1
            # print 'mask1.shape',mask1.shape
            # print 'length',length
            test = T.neq(self.y_pred, y).reshape([n_sample,batch_size/n_sample])[T.arange(n_sample), length]
            # test = T.neq(self.y_pred, y)[length]
            # print 'test.shape',test.shape

            # return T.mean(T.neq(self.y_pred, y), dtype=config.floatX)
            return T.mean(test, dtype=config.floatX)
            # return test.astype(config.floatX)
        else:
            raise NotImplementedError()
