from theano import config
import numpy
from collections import OrderedDict
import theano
import theano.tensor as tensor
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj
def _p(pp, name):
    return '%s_%s' % (pp, name)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)
def uniform_weight(n_in,n_out):
    rng = numpy.random.RandomState(1234)
    W= numpy.asarray(
                rng.uniform(
                    low = -0.08,
                    high = 0.08,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
    )
    return W
def lstm_param_init(lstm_options):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    U_modal = numpy.concatenate([uniform_weight(lstm_options['dim_proj'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_proj'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_proj'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_proj'],lstm_options['dim_proj'])], axis=1)
    U_modal_vid = numpy.concatenate([uniform_weight(lstm_options['dim_flow'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_flow'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_flow'],lstm_options['dim_proj']),
                       uniform_weight(lstm_options['dim_flow'],lstm_options['dim_proj'])], axis=1)
    LSTM_U_modal =theano.shared(value=U_modal)
    LSTM_W_modal =theano.shared(value=U_modal_vid)

    b_modal = numpy.zeros((4 * lstm_options['dim_proj'],)).astype(config.floatX)
    LSTM_b_modal = theano.shared(value=b_modal)
    conv_dim =lstm_options['conv_dim']
    dim_part =lstm_options['dim_part']
    dim_w = dim_part*5#512
   
    LSTM_W_lamada_modal =uniform_weight(dim_w/5,lstm_options['num_joints'])  #numpy.zeros((50,1)).astype(config.floatX)
    LSTM_U_lamada_modal =uniform_weight(conv_dim,dim_w)  #numpy.zeros((512,50)).astype(config.floatX)
    LSTM_H_lamada_modal =uniform_weight(lstm_options['dim_proj'],dim_w)  #numpy.zeros((1024,50)).astype(config.floatX)
    LSTM_b_lamada_modal = numpy.zeros((dim_w,)).astype(config.floatX)

    LSTM_W_lamada_modal = theano.shared(value= LSTM_W_lamada_modal)
    LSTM_U_lamada_modal =theano.shared(value= LSTM_U_lamada_modal)
    LSTM_H_lamada_modal =theano.shared(value= LSTM_H_lamada_modal)
    LSTM_b_lamada_modal = theano.shared(value=LSTM_b_lamada_modal)

    return LSTM_H_lamada_modal,LSTM_U_lamada_modal,LSTM_W_lamada_modal,LSTM_b_lamada_modal,LSTM_U_modal, LSTM_W_modal,LSTM_b_modal

def joint_attention_lstm(conv_fea,LSTM_H_lamada_modal,LSTM_U_lamada_modal,LSTM_W_lamada_modal,LSTM_b_lamada_modal,
                             LSTM_U_modal,LSTM_W_modal,LSTM_b_modal,
                             lstm_options, prefix='lstm',mask=None):
    nsteps=lstm_options['n_timesteps']
    num_seq =lstm_options['num_seq']
    assert mask is not None
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    limbs_up_elbow = [5,6]
    limbs_up_wrist = [9, 10]
    limbs_down_knee = [7, 8]
    limbs_down_ankle = [ 11, 12]
    others = [0, 1, 2, 3, 4] #5,6,9,10,7,8,11,12,0,1,2,3,4
    # raw_idx =[8,9,10,0,1,2,3,11,12,4,5,6,7]
    raw_idx =[8,9,10,11,12,0,1,4,5,2,3,6,7]
    #    5,6,9,10,7,8,11,12,0,1, 2, 3, 4
    # -->0,1,2, 3,4,5, 6, 7,8,9,10,11,12
    # tensor.concatenate([h_modal_pool4,h_tmp_pool4],axis=1)
    dim_part =lstm_options['dim_part']#dim_w/5
    # dim_w=dim_part*5
    def _step(m_, range_num_timesteps_shared, h_modal_, c_modal_, lamada_com):
        conv_fea_att_step_tmp = conv_fea[:, range_num_timesteps_shared, :, :]  # [:,:,:] #49*16*n_sample*512
        lamda_modal =tensor.tanh(tensor.dot(conv_fea_att_step_tmp, LSTM_U_lamada_modal) +  # 49*n_sample*512
                       tensor.dot(h_modal_, LSTM_H_lamada_modal)[None, :, :] + LSTM_b_lamada_modal)#,
        lamda_modal_up_elbow = tensor.dot(_slice(lamda_modal, 0, dim_part),LSTM_W_lamada_modal[:,:2])
        lamda_modal_up_wrist = tensor.dot(_slice(lamda_modal, 1, dim_part),LSTM_W_lamada_modal[:,2:4])
        lamda_modal_down_knee = tensor.dot(_slice(lamda_modal, 2, dim_part),LSTM_W_lamada_modal[:,4:6])
        lamda_modal_down_ankle = tensor.dot(_slice(lamda_modal, 3, dim_part),LSTM_W_lamada_modal[:,6:8])
        lamda_modal_others = tensor.dot(_slice(lamda_modal, 4, dim_part),LSTM_W_lamada_modal[:,8:])
        lamda_modal_parts=tensor.concatenate([lamda_modal_up_elbow,lamda_modal_up_wrist,lamda_modal_down_knee,lamda_modal_down_ankle,lamda_modal_others],axis=2)

        lamda_modal_parts = (numpy.exp(lamda_modal_parts)) / ((numpy.exp(lamda_modal_parts)).sum(axis=0))
        lamada_com = lamda_modal_parts[:,:,raw_idx]  # [:,:,0]
        # lamada_com_pool5 = lamada_com  # 49 * batch_size*num_joints
        h_tmp_pool4 = conv_fea_att_step_tmp[:, :, None, :] * lamada_com[:, :, :, None]  # 49* batchsize*13*51

        h_tmp_pool4 = tensor.sum(h_tmp_pool4, axis=0)  # batchsize* 13* 512
        h_tmp_pool4_up_left = tensor.sum(h_tmp_pool4[:, limbs_up_elbow, :], axis=1)  # batchsize*  512
        h_tmp_pool4_up_right = tensor.sum(h_tmp_pool4[:, limbs_up_wrist, :], axis=1)  # batchsize*  512
        h_tmp_pool4_down_knee = tensor.sum(h_tmp_pool4[:, limbs_down_knee, :], axis=1)  # batchsize*  512
        h_tmp_pool4_down_ankle = tensor.sum(h_tmp_pool4[:, limbs_down_ankle, :], axis=1)  # batchsize*  512
        h_tmp_pool4_others = tensor.sum(h_tmp_pool4[:, others, :], axis=1)  # batchsize*  512

        h_joint = tensor.concatenate([h_tmp_pool4_up_left,h_tmp_pool4_up_right,h_tmp_pool4_down_knee,h_tmp_pool4_down_ankle, h_tmp_pool4_others],
                                     axis=1)
        preact_modal = tensor.dot(h_modal_, LSTM_U_modal).astype(config.floatX)
        preact_modal +=tensor.dot(h_joint, LSTM_W_modal)#.astype(config.floatX)
        preact_modal += LSTM_b_modal

        i_modal = tensor.nnet.sigmoid(_slice(preact_modal, 0, lstm_options['dim_proj']))
        f_modal = tensor.nnet.sigmoid(_slice(preact_modal, 1, lstm_options['dim_proj']))
        o_modal = tensor.nnet.sigmoid(_slice(preact_modal, 2, lstm_options['dim_proj']))
        c_modal = tensor.tanh(_slice(preact_modal, 3, lstm_options['dim_proj']))

        c_modal = f_modal * c_modal_ + i_modal * c_modal
        c_modal = m_[:, None] * c_modal + (1. - m_)[:, None] * c_modal_

        h_modal = o_modal * tensor.tanh(c_modal)
        h_modal = m_[:, None] * h_modal + (1. - m_)[:, None] * h_modal_
        h_modal = h_modal.astype(config.floatX)
        c_modal = c_modal.astype(config.floatX)
        #################
        return h_modal, c_modal,lamada_com

    dim_proj = lstm_options['dim_proj']
    num_joints = lstm_options['num_joints']
    reg_x = lstm_options['reg_scale_x']
    reg_y = lstm_options['reg_scale_y']
    range_num_timesteps = numpy.asarray(range(lstm_options['n_timesteps']))
    range_num_timesteps_shared = theano.shared(range_num_timesteps)
    rval, updates = theano.scan(_step,
                                sequences=[mask,range_num_timesteps_shared],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           num_seq,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           num_seq,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           reg_x*reg_y+1,
                                                           num_seq,
                                                           num_joints)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0],rval[2]


