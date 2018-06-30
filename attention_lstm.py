import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
# theano.config.warn_float64='raise'
import theano.tensor as T

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from layers import SoftmaxLayer,JointAttentionLstmLayer
from lstm_lib import numpy_floatX


class LSTM_softmax(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        num_seq = config['num_seq']
        self.n_timesteps = config['num_timesteps']
        
        num_joints = config['num_joints']
        classes_num =config['classes_num']
        # ##################### BUILD NETWORK ##########################
        mask = T.fvector('mask')
        y = T.lvector('y')
        target = T.ftensor3('target')
        rand = T.fvector('rand')
        trng = RandomStreams(1234)
        use_noise=T.fscalar('use_noise')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        conv_fea = T.ftensor4('conv_fea')#(49, 16, 8, 1024)


        lstm_att_layer15 = JointAttentionLstmLayer(  config,num_joints,conv_fea=conv_fea,
                                                   mask=mask, batch_size=batch_size,num_seq=num_seq, trng=trng,
                                use_noise=use_noise, n_in=1024*5, n_out=1024,dim_part=32)

        self.layers.append(lstm_att_layer15)
        params += lstm_att_layer15.params
        weight_types += lstm_att_layer15.weight_type
        self.conv_fea = conv_fea

        softmax_input=lstm_att_layer15.output

        softmax_layer15 = SoftmaxLayer(
            input=softmax_input, n_in=1024, n_out=21)
        self.layers.append(softmax_layer15)
        params += softmax_layer15.params
        weight_types += softmax_layer15.weight_type

        # #################### NETWORK BUILT #######################
        self.cost_nll = softmax_layer15.negative_log_likelihood(y,mask)
        self.cost_jhmdb_attention =T.mean(T.sum(T.sum(0.5*(lstm_att_layer15.attention-target)**2,axis=1),axis=1),axis=0,dtype=theano.config.floatX)
        self.cost=self.cost_nll+self.cost_jhmdb_attention
        self.errors_video =softmax_layer15.errors_video( y, mask, batch_size,num_seq)
        self.params = params
        self.prob = softmax_layer15.p_y_given_x
        
        self.mask =mask
        self.y = y
        self.target =target
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.num_seq = num_seq
        self.use_noise=use_noise


def compile_models_LSTM(model, config, flag_top_5=False):

    conv_fea = model.conv_fea
    mask = model.mask
    y = model.y
    target=model.target
    rand = model.rand
    num_joints = config['num_joints']
    weight_types = model.weight_types
    use_noise=model.use_noise
    cost = model.cost
    params = model.params
    reg_scale_x = config['reg_scale_x']
    reg_scale_y = config['reg_scale_y']
    errors_video=model.errors_video
    batch_size = model.batch_size
    num_seq=model.num_seq
    n_timesteps = model.n_timesteps
    cost_nll = model.cost_nll
    cost_att = model.cost_jhmdb_attention
    prob = model.prob
    mu = config['momentum']
    
    eta = config['weight_decay']

    assert len(weight_types) == len(params)
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))
    lr = T.scalar('lr')  # symbolic learning rate
   
    shared_conv_fea = theano.shared(np.zeros((reg_scale_x*reg_scale_y+1, n_timesteps, num_seq,
                                       1024),
                                  dtype=theano.config.floatX),
                         borrow=True)
    shared_mask = theano.shared(np.zeros((batch_size,),
                                         dtype=theano.config.floatX),
                                borrow=True)

    shared_y = theano.shared(np.zeros((batch_size,), dtype=int),
                             borrow=True)
    shared_target = theano.shared(np.zeros((batch_size,reg_scale_x*reg_scale_y+1,num_joints), dtype=theano.config.floatX),
                             borrow=True)
    rand_arr = theano.shared(np.zeros(3, dtype=theano.config.floatX),
                             borrow=True)
    shared_use_noise = theano.shared(numpy_floatX(0.),borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]
    if config['use_momentum']:
        print 'params for training:',len(weight_types),len(params)
        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type =='W_lr':
                real_grad = grad_i + eta * param_i
                real_lr = 0.1*lr

            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            elif weight_type == 'b_lr':
                real_grad = grad_i
                real_lr = 0.1*2. * lr
            else:
                raise TypeError("Weight Type Error")

            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")
    # Define Theano Functions
    train_model = theano.function([], [cost,cost_nll,cost_att], updates=updates,
                                  givens=[
                                      (conv_fea,shared_conv_fea),
                                          (mask, shared_mask),
                                          (y, shared_y),
                                      (target,shared_target),
                                          (lr, learning_rate),
                                          (rand, rand_arr),
                                          (use_noise,shared_use_noise)])
    if config['phase']=='test':
        validate_outputs =[cost_nll,errors_video,prob]
    else:
        validate_outputs =[cost_nll,errors_video]
    

    validate_model = theano.function([], validate_outputs,
                                     givens=[(conv_fea,shared_conv_fea),
                                             (mask, shared_mask),(y, shared_y),
                                             (rand, rand_arr),
                                             (use_noise,shared_use_noise)])
    train_error = theano.function([], errors_video, givens=[
            (conv_fea,shared_conv_fea),
             (mask, shared_mask), (y, shared_y),(target,shared_target),(rand, rand_arr),
            (use_noise,shared_use_noise)])
    return (train_model, validate_model, train_error,
            learning_rate, shared_conv_fea, shared_mask, shared_y,shared_target, rand_arr,shared_use_noise, vels)
