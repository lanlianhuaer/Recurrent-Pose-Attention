import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
# theano.config.warn_float64='raise'
import theano.tensor as T

import numpy as np
from layers import ConvPoolLayer, PoolLayer


class InceptionBN(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        num_seq = config['num_seq']
        lib_conv = config['lib_conv']

        # ##################### BUILD NETWORK ##########################
        img_scale_x = config['img_scale_x']
        img_scale_y = config['img_scale_y']
        reg_scale_x = config['reg_scale_x']
        reg_scale_y = config['reg_scale_y']
        use_noise=T.fscalar('use_noise')
        input_dim = config['input_dim']
        print '... building the model'
        self.layers = []
        params = []
        weight_types = []
        x_temporal = T.ftensor4('x')


        conv1_temporal = ConvPoolLayer(input=x_temporal,
                                        image_shape=(input_dim, img_scale_x, img_scale_y, batch_size),
                                        filter_shape=(input_dim, 7, 7, 64),
                                        convstride=2, padsize=3, group=1,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, lrn=False,Bn=True,
                                        lib_conv=lib_conv,caffe_style=True,poolpadsize=(1,1)
                                        )
        self.layers.append(conv1_temporal)

        conv_temporal_2_reduce = ConvPoolLayer(input=conv1_temporal.output,
                                        image_shape=(64, 56, 56, batch_size),
                                        filter_shape=(64, 1, 1, 64),
                                        convstride=1, padsize=0, group=1,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, lrn=False,Bn=True,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(conv_temporal_2_reduce)
        #
        convpool_temporal_2 = ConvPoolLayer(input=conv_temporal_2_reduce.output,
                                        image_shape=(64, 56, 56, batch_size),
                                        filter_shape=(64, 3, 3, 192),
                                        convstride=1, padsize=1, group=1,
                                        poolsize=3, poolstride=2,#poolpadsize=(1,1),
                                        bias_init=0.0, lrn=False,Bn=True,
                                        lib_conv=lib_conv,caffe_style=True,poolpadsize=(1,1)
                                        )
        self.layers.append(convpool_temporal_2)

        ##############----3a---#########

        inception_temporal_3a_1x1=ConvPoolLayer(input=convpool_temporal_2.output,image_shape=(192,28,28,batch_size),
                                        filter_shape=(192,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_1x1)
        #################
        inception_temporal_3a_3x3_reduce=ConvPoolLayer(input=convpool_temporal_2.output,image_shape=(192,28,28,batch_size),
                                        filter_shape=(192,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_3x3_reduce)
        inception_temporal_3a_3x3=ConvPoolLayer(input=inception_temporal_3a_3x3_reduce.output,image_shape=(64,28,28,batch_size),
                                        filter_shape=(64,3,3,64),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_3x3)
        ############
        inception_temporal_3a_double_3x3_reduce=ConvPoolLayer(input=convpool_temporal_2.output,image_shape=(192,28,28,batch_size),
                                        filter_shape=(192,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_double_3x3_reduce)
        inception_temporal_3a_double_3x3_1=ConvPoolLayer(input=inception_temporal_3a_double_3x3_reduce.output,image_shape=(64,28,28,batch_size),
                                        filter_shape=(64,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_double_3x3_1)
        inception_temporal_3a_double_3x3_2=ConvPoolLayer(input=inception_temporal_3a_double_3x3_1.output,image_shape=(96,28,28,batch_size),
                                        filter_shape=(96,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_double_3x3_2)
        ##############
        inception_temporal_3a_pool=PoolLayer(input=convpool_temporal_2.output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_3a_pool_proj=ConvPoolLayer(input=inception_temporal_3a_pool.output,image_shape=(192,28,28,batch_size),
                                        filter_shape=(192,1,1,32),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3a_pool_proj)

        ####################
        inception_temporal_3a_output=T.concatenate([inception_temporal_3a_1x1.output,inception_temporal_3a_3x3.output,inception_temporal_3a_double_3x3_2.output,
                                           inception_temporal_3a_pool_proj.output],
                                          axis=0)

        ##############----3b---#########
        inception_temporal_3b_1x1=ConvPoolLayer(input=inception_temporal_3a_output,image_shape=(256,28,28,batch_size),
                                       filter_shape=(256,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_1x1)
        #######################
        inception_temporal_3b_3x3_reduce=ConvPoolLayer(input=inception_temporal_3a_output,image_shape=(256,28,28,batch_size),
                                        filter_shape=(256,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_3x3_reduce)
        inception_temporal_3b_3x3=ConvPoolLayer(input=inception_temporal_3b_3x3_reduce.output,image_shape=(64,28,28,batch_size),
                                        filter_shape=(64,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_3x3)

        ############
        inception_temporal_3b_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_3a_output,image_shape=(256,28,28,batch_size),
                                        filter_shape=(256,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_double_3x3_reduce)
        inception_temporal_3b_double_3x3_1=ConvPoolLayer(input=inception_temporal_3b_double_3x3_reduce.output,image_shape=(64,28,28,batch_size),
                                        filter_shape=(64,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_double_3x3_1)
        inception_temporal_3b_double_3x3_2=ConvPoolLayer(input=inception_temporal_3b_double_3x3_1.output,image_shape=(96,28,28,batch_size),
                                        filter_shape=(96,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_double_3x3_2)
        ##############
        inception_temporal_3b_pool=PoolLayer(input=inception_temporal_3a_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_3b_pool_proj=ConvPoolLayer(input=inception_temporal_3b_pool.output,image_shape=(256,28,28,batch_size),
                                        filter_shape=(256,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3b_pool_proj)
        ###############33

        inception_temporal_3b_output=T.concatenate([inception_temporal_3b_1x1.output,inception_temporal_3b_3x3.output,inception_temporal_3b_double_3x3_2.output,
                                           inception_temporal_3b_pool_proj.output],axis=0)

        ##############----3c---#########
        inception_temporal_3c_3x3_reduce=ConvPoolLayer(input=inception_temporal_3b_output,image_shape=(320,28,28,batch_size),
                                        filter_shape=(320,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3c_3x3_reduce)
        inception_temporal_3c_3x3=ConvPoolLayer(input=inception_temporal_3c_3x3_reduce.output,image_shape=(128,28,28,batch_size),
                                        filter_shape=(128,3,3,160),convstride=2,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3c_3x3)
        ############
        inception_temporal_3c_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_3b_output,image_shape=(320,28,28,batch_size),
                                        filter_shape=(320,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3c_double_3x3_reduce)
        inception_temporal_3c_double_3x3_1=ConvPoolLayer(input=inception_temporal_3c_double_3x3_reduce.output,image_shape=(64,28,28,batch_size),
                                        filter_shape=(64,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3c_double_3x3_1)
        inception_temporal_3c_double_3x3_2=ConvPoolLayer(input=inception_temporal_3c_double_3x3_1.output,image_shape=(96,28,28,batch_size),
                                        filter_shape=(96,3,3,96),convstride=2,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_3c_double_3x3_2)
        ##############
        inception_temporal_3c_pool=PoolLayer(input=inception_temporal_3b_output,poolsize=3,poolstride=2,lib_conv=lib_conv,
                                    caffe_style=True,poolpad=1)
        # inception_temporal_3c_pool=PoolLayer(input=inception_temporal_3b_output,caffe_style=True,poolsize=3,poolpad=1,poolstride=2,lib_conv=lib_conv)
        #################################
        inception_temporal_3c_output=T.concatenate([inception_temporal_3c_3x3.output,inception_temporal_3c_double_3x3_2.output,
                                           inception_temporal_3c_pool.output],axis=0)


        ################################----4a------##########
        inception_temporal_4a_1x1=ConvPoolLayer(input=inception_temporal_3c_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,224),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_1x1)
        #################
        inception_temporal_4a_3x3_reduce=ConvPoolLayer(input=inception_temporal_3c_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,64),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_3x3_reduce)
        inception_temporal_4a_3x3=ConvPoolLayer(input=inception_temporal_4a_3x3_reduce.output,image_shape=(64,14,14,batch_size),
                                        filter_shape=(64,3,3,96),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_3x3)
        ############
        inception_temporal_4a_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_3c_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,96),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_double_3x3_reduce)
        inception_temporal_4a_double_3x3_1=ConvPoolLayer(input=inception_temporal_4a_double_3x3_reduce.output,image_shape=(96,14,14,batch_size),
                                        filter_shape=(96,3,3,128),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_double_3x3_1)
        inception_temporal_4a_double_3x3_2=ConvPoolLayer(input=inception_temporal_4a_double_3x3_1.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,128),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_double_3x3_2)
        ##############
        inception_temporal_4a_pool=PoolLayer(input=inception_temporal_3c_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_4a_pool_proj=ConvPoolLayer(input=inception_temporal_4a_pool.output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4a_pool_proj)

        ####################
        inception_temporal_4a_output=T.concatenate([inception_temporal_4a_1x1.output,inception_temporal_4a_3x3.output,
                                           inception_temporal_4a_double_3x3_2.output,inception_temporal_4a_pool_proj.output],
                                          axis=0)
        #####################----4b------#################
        inception_temporal_4b_1x1=ConvPoolLayer(input=inception_temporal_4a_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,192),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_1x1)
        #################
        inception_temporal_4b_3x3_reduce=ConvPoolLayer(input=inception_temporal_4a_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,96),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_3x3_reduce)
        inception_temporal_4b_3x3=ConvPoolLayer(input=inception_temporal_4b_3x3_reduce.output,image_shape=(96,14,14,batch_size),
                                        filter_shape=(96,3,3,128),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_3x3)
        ############
        inception_temporal_4b_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_4a_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,96),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_double_3x3_reduce)
        inception_temporal_4b_double_3x3_1=ConvPoolLayer(input=inception_temporal_4b_double_3x3_reduce.output,image_shape=(96,14,14,batch_size),
                                        filter_shape=(96,3,3,128),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_double_3x3_1)
        inception_temporal_4b_double_3x3_2=ConvPoolLayer(input=inception_temporal_4b_double_3x3_1.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,128),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_double_3x3_2)
        ##############
        inception_temporal_4b_pool=PoolLayer(input=inception_temporal_4a_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_4b_pool_proj=ConvPoolLayer(input=inception_temporal_4b_pool.output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4b_pool_proj)

        ####################
        inception_temporal_4b_output=T.concatenate([inception_temporal_4b_1x1.output,inception_temporal_4b_3x3.output,
                                           inception_temporal_4b_double_3x3_2.output,inception_temporal_4b_pool_proj.output],
                                          axis=0)
        #####################----4c------#################
        inception_temporal_4c_1x1=ConvPoolLayer(input=inception_temporal_4b_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,160),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_1x1)
        #################
        inception_temporal_4c_3x3_reduce=ConvPoolLayer(input=inception_temporal_4b_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_3x3_reduce)
        inception_temporal_4c_3x3=ConvPoolLayer(input=inception_temporal_4c_3x3_reduce.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,160),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_3x3)
        ############
        inception_temporal_4c_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_4b_output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_double_3x3_reduce)
        inception_temporal_4c_double_3x3_1=ConvPoolLayer(input=inception_temporal_4c_double_3x3_reduce.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,160),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_double_3x3_1)
        inception_temporal_4c_double_3x3_2=ConvPoolLayer(input=inception_temporal_4c_double_3x3_1.output,image_shape=(160,14,14,batch_size),
                                        filter_shape=(160,3,3,160),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_double_3x3_2)
        ##############
        inception_temporal_4c_pool=PoolLayer(input=inception_temporal_4b_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_4c_pool_proj=ConvPoolLayer(input=inception_temporal_4c_pool.output,image_shape=(576,14,14,batch_size),
                                        filter_shape=(576,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4c_pool_proj)

        ####################
        inception_temporal_4c_output=T.concatenate([inception_temporal_4c_1x1.output,inception_temporal_4c_3x3.output,
                                           inception_temporal_4c_double_3x3_2.output,inception_temporal_4c_pool_proj.output],
                                          axis=0)

        #####################----4d------#################
        inception_temporal_4d_1x1=ConvPoolLayer(input=inception_temporal_4c_output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,96),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_1x1)
        #################
        inception_temporal_4d_3x3_reduce=ConvPoolLayer(input=inception_temporal_4c_output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_3x3_reduce)
        inception_temporal_4d_3x3=ConvPoolLayer(input=inception_temporal_4d_3x3_reduce.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,192),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_3x3)
        ############
        inception_temporal_4d_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_4c_output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,160),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_double_3x3_reduce)
        inception_temporal_4d_double_3x3_1=ConvPoolLayer(input=inception_temporal_4d_double_3x3_reduce.output,image_shape=(160,14,14,batch_size),
                                        filter_shape=(160,3,3,192),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_double_3x3_1)
        inception_temporal_4d_double_3x3_2=ConvPoolLayer(input=inception_temporal_4d_double_3x3_1.output,image_shape=(192,14,14,batch_size),
                                        filter_shape=(192,3,3,192),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_double_3x3_2)
        ##############
        inception_temporal_4d_pool=PoolLayer(input=inception_temporal_4c_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_4d_pool_proj=ConvPoolLayer(input=inception_temporal_4d_pool.output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4d_pool_proj)

        ####################
        inception_temporal_4d_output=T.concatenate([inception_temporal_4d_1x1.output,inception_temporal_4d_3x3.output,
                                           inception_temporal_4d_double_3x3_2.output,inception_temporal_4d_pool_proj.output],
                                          axis=0)

        ##############----4e---#########


        inception_temporal_4e_3x3_reduce=ConvPoolLayer(input=inception_temporal_4d_output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4e_3x3_reduce)
        inception_temporal_4e_3x3=ConvPoolLayer(input=inception_temporal_4e_3x3_reduce.output,image_shape=(128,14,14,batch_size),
                                        filter_shape=(128,3,3,192),convstride=2,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4e_3x3)
        ############
        inception_temporal_4e_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_4d_output,image_shape=(608,14,14,batch_size),
                                        filter_shape=(608,1,1,192),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4e_double_3x3_reduce)
        inception_temporal_4e_double_3x3_1=ConvPoolLayer(input=inception_temporal_4e_double_3x3_reduce.output,image_shape=(192,14,14,batch_size),
                                        filter_shape=(192,3,3,256),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4e_double_3x3_1)
        inception_temporal_4e_double_3x3_2=ConvPoolLayer(input=inception_temporal_4e_double_3x3_1.output,image_shape=(256,14,14,batch_size),
                                        filter_shape=(256,3,3,256),convstride=2,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_4e_double_3x3_2)
        ##############
        inception_temporal_4e_pool=PoolLayer(input=inception_temporal_4d_output,poolsize=3,poolstride=2,lib_conv=lib_conv,
                                    caffe_style=True,poolpad=1)
        #################################
        inception_temporal_4e_output=T.concatenate([inception_temporal_4e_3x3.output,inception_temporal_4e_double_3x3_2.output,
                                           inception_temporal_4e_pool.output],axis=0)
        ################################----5a------##########
        inception_temporal_5a_1x1=ConvPoolLayer(input=inception_temporal_4e_output,image_shape=(1056,7,7,batch_size),
                                        filter_shape=(1056,1,1,352),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_1x1)
        #################
        inception_temporal_5a_3x3_reduce=ConvPoolLayer(input=inception_temporal_4e_output,image_shape=(1056,7,7,batch_size),
                                        filter_shape=(1056,1,1,192),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_3x3_reduce)
        inception_temporal_5a_3x3=ConvPoolLayer(input=inception_temporal_5a_3x3_reduce.output,image_shape=(192,7,7,batch_size),
                                        filter_shape=(192,3,3,320),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_3x3)
        ############
        inception_temporal_5a_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_4e_output,image_shape=(1056,7,7,batch_size),
                                        filter_shape=(1056,1,1,160),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_double_3x3_reduce)
        inception_temporal_5a_double_3x3_1=ConvPoolLayer(input=inception_temporal_5a_double_3x3_reduce.output,image_shape=(160,7,7,batch_size),
                                        filter_shape=(160,3,3,224),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_double_3x3_1)
        inception_temporal_5a_double_3x3_2=ConvPoolLayer(input=inception_temporal_5a_double_3x3_1.output,image_shape=(224,7,7,batch_size),
                                        filter_shape=(224,3,3,224),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_double_3x3_2)
        ##############
        inception_temporal_5a_pool=PoolLayer(input=inception_temporal_4e_output,poolsize=3,poolstride=1,poolpad=1,poolmode='average_inc_pad',lib_conv=lib_conv)
        inception_temporal_5a_pool_proj=ConvPoolLayer(input=inception_temporal_5a_pool.output,image_shape=(1056,7,7,batch_size),
                                        filter_shape=(1056,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5a_pool_proj)

        ####################
        inception_temporal_5a_output=T.concatenate([inception_temporal_5a_1x1.output,inception_temporal_5a_3x3.output,
                                           inception_temporal_5a_double_3x3_2.output,inception_temporal_5a_pool_proj.output]
                                          ,axis=0)

        inception_temporal_5a_output_1 =inception_temporal_5a_output
        ################################----5b------##########
        inception_temporal_5b_1x1=ConvPoolLayer(input=inception_temporal_5a_output,image_shape=(1024,7,7,batch_size),
                                        filter_shape=(1024,1,1,352),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_1x1)
        #################
        inception_temporal_5b_3x3_reduce=ConvPoolLayer(input=inception_temporal_5a_output,image_shape=(1024,7,7,batch_size),
                                        filter_shape=(1024,1,1,192),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_3x3_reduce)
        inception_temporal_5b_3x3=ConvPoolLayer(input=inception_temporal_5b_3x3_reduce.output,image_shape=(192,7,7,batch_size),
                                        filter_shape=(192,3,3,320),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_3x3)
        ############
        inception_temporal_5b_double_3x3_reduce=ConvPoolLayer(input=inception_temporal_5a_output,image_shape=(1024,7,7,batch_size),
                                        filter_shape=(1024,1,1,192),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_double_3x3_reduce)
        inception_temporal_5b_double_3x3_1=ConvPoolLayer(input=inception_temporal_5b_double_3x3_reduce.output,image_shape=(192,7,7,batch_size),
                                        filter_shape=(192,3,3,224),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_double_3x3_1)
        inception_temporal_5b_double_3x3_2=ConvPoolLayer(input=inception_temporal_5b_double_3x3_1.output,image_shape=(224,7,7,batch_size),
                                        filter_shape=(224,3,3,224),convstride=1,padsize=1,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_double_3x3_2)
        ##############
        inception_temporal_5b_pool=PoolLayer(input=inception_temporal_5a_output,poolsize=3,poolstride=1,poolpad=1,lib_conv=lib_conv)
        inception_temporal_5b_pool_proj=ConvPoolLayer(input=inception_temporal_5b_pool.output,image_shape=(1024,7,7,batch_size),
                                        filter_shape=(1024,1,1,128),convstride=1,padsize=0,group=1,
                                       poolsize=1,poolstride=1,
                                       bias_init=0.0,lib_conv=lib_conv,Bn=True)
        self.layers.append(inception_temporal_5b_pool_proj)
        #params += inception_temporal_5b_pool_proj.params
       # weight_types += inception_temporal_5b_pool_proj.weight_type

        ####################
        dummy_fea = T.zeros([1024,1,num_seq,batch_size/num_seq])
        pool5_fea_tmp=T.reshape(inception_temporal_5a_output_1,[1024,reg_scale_x*reg_scale_y,num_seq,batch_size/num_seq])
        pool5_fea_tmp = T.concatenate([pool5_fea_tmp,dummy_fea],axis=1)
        pool5_fea_tmp = pool5_fea_tmp.dimshuffle(1,3,2,0)

        self.fea_tmp = pool5_fea_tmp
        # self.fea_lstm_tmp = pool5_fea_tmp
        self.params = params
        self.x_temporal = x_temporal
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.num_seq = num_seq
        self.use_noise=use_noise

def compile_models(model, config, flag_top_5=False):

    x_temporal = model.x_temporal
    weight_types = model.weight_types
    params = model.params
    batch_size = model.batch_size
    img_scale_x = config['img_scale_x']
    img_scale_y = config['img_scale_y']
    imput_dim =config['input_dim']

    assert len(weight_types) == len(params)

    shared_x_temporal = theano.shared(np.zeros((imput_dim, img_scale_x, img_scale_y,
                                   batch_size),
                                  dtype=theano.config.floatX),
                         borrow=True)

    train_model = theano.function([], model.fea_tmp,
                                  givens=[
                                      (x_temporal,shared_x_temporal)])
    return (train_model,  shared_x_temporal)
