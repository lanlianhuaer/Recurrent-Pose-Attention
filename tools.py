import os

import numpy as np
import theano


def save_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.save_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.save_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.save_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.save_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.save_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.save_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'gamma'):

            name ='gamma' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].gamma.get_value())

        if hasattr(layers[idx], 'beta'):
            name ='beta' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].beta.get_value())
        if hasattr(layers[idx], 'mean'):
            name ='mean' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].mean.get_value())
        if hasattr(layers[idx], 'var'):
            name ='var' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].var.get_value())
        if hasattr(layers[idx], 'LSTM_H_lamada_modal'):
            name = 'LSTM_H_lamada_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_H_lamada_modal.get_value())
        if hasattr(layers[idx], 'LSTM_H_lamada_part'):
            name = 'LSTM_H_lamada_part' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_H_lamada_part.get_value())
        if hasattr(layers[idx], 'LSTM_W_lamada_part'):
            name = 'LSTM_W_lamada_part' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_W_lamada_part.get_value())
        if hasattr(layers[idx], 'LSTM_U_lamada_modal'):
            name = 'LSTM_U_lamada_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_U_lamada_modal.get_value())
        if hasattr(layers[idx], 'LSTM_W_lamada_modal'):
            name = 'LSTM_W_lamada_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_W_lamada_modal.get_value())
        if hasattr(layers[idx], 'LSTM_b_lamada_modal'):
            name = 'LSTM_b_lamada_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_b_lamada_modal.get_value())
        if hasattr(layers[idx], 'LSTM_W_modal'):
            name = 'LSTM_W_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_W_modal.get_value())
        if hasattr(layers[idx], 'LSTM_U_modal'):
            name = 'LSTM_U_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_U_modal.get_value())
        if hasattr(layers[idx], 'LSTM_b_modal'):
            name = 'LSTM_b_modal' + '_' + str(idx) + '_' + str(epoch)
            np.save(weights_dir + name + '.npy', layers[idx].LSTM_b_modal.get_value())
    print('weight saved')
def load_weights_inception(layers,idx,epoch,weights_dir,lstm_param,lstm_param_name):
    name=lstm_param_name+ '_' + str(idx) + '_' + str(epoch)
    if os.path.exists(weights_dir + name + '.npy'):
        np_values = np.load(weights_dir + name + '.npy').astype('float32')
        if np_values.shape==lstm_param.get_value().shape:
            # print(weights_dir,name)
            # np.load(weights_dir + name + '.npy').astype('float32')
            lstm_param.set_value(np_values)
            print 'weight loaded:2 ' , name ,', shape:' , np_values.shape
        elif np_values.shape[1]==lstm_param.get_value().shape[0]:
            #gamma.shape 1,*,1,1   ---->(*,)
            np_values= np_values[0,:,0,0]
            lstm_param.set_value(np_values)
            print 'weight loaded:2 ' , name ,', shape:' , np_values.shape
        else:
            print('dim not match from', np_values.shape, 'to',lstm_param.get_value().shape)
        # print('right')
    else:
        print 'warning: weight'+ name + ' not found,ignored... '
def load_weights_lstm(layers,idx,epoch,weights_dir,lstm_param,lstm_param_name):
    name=lstm_param_name+ '_' + str(idx) + '_' + str(epoch)
    if os.path.exists(weights_dir + name + '.npy'):
        np_values = np.load(weights_dir + name + '.npy').astype('float32')
        if np_values.shape==lstm_param.get_value().shape:
            lstm_param.set_value(np_values)
            print 'weight loaded: ' , name ,', shape:' , np_values.shape
        else:
            print('dim not match from', np_values.shape, 'to',lstm_param.get_value().shape)
    else:
        print 'warning: weight'+ name + ' not found,ignored... '

def load_weights(layers, weights_dir, epoch,gap):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx+gap) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx+gap) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx+gap) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx+gap) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight(
                weights_dir, 'b0' + '_' + str(idx+gap) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight(
                weights_dir, 'b1' + '_' + str(idx+gap) + '_' + str(epoch))
        idx_gap= idx+gap
        if hasattr(layers[idx], 'mean'):
            load_weights_lstm(layers,idx_gap,epoch,weights_dir,layers[idx].mean,'mean')
        if hasattr(layers[idx], 'var'):
            load_weights_lstm(layers,idx_gap,epoch,weights_dir,layers[idx].var,'var')
        if hasattr(layers[idx], 'beta'):
            load_weights_lstm(layers,idx_gap,epoch,weights_dir,layers[idx].beta,'beta')
        if hasattr(layers[idx], 'gamma'):
            load_weights_lstm(layers,idx_gap,epoch,weights_dir,layers[idx].gamma,'gamma')
        if hasattr(layers[idx], 'LSTM_H_lamada_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_H_lamada_modal,'LSTM_H_lamada_modal')
        if hasattr(layers[idx], 'LSTM_H_lamada_part'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_H_lamada_part,'LSTM_H_lamada_part')
        if hasattr(layers[idx], 'LSTM_W_lamada_part'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_W_lamada_part,'LSTM_W_lamada_part')
        if hasattr(layers[idx], 'LSTM_U_lamada_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_U_lamada_modal,'LSTM_U_lamada_modal')
        if hasattr(layers[idx], 'LSTM_W_lamada_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_W_lamada_modal,'LSTM_W_lamada_modal')
        if hasattr(layers[idx], 'LSTM_b_lamada_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_b_lamada_modal,'LSTM_b_lamada_modal')
        if hasattr(layers[idx], 'LSTM_W_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_W_modal,'LSTM_W_modal')
        if hasattr(layers[idx], 'LSTM_U_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_U_modal,'LSTM_U_modal')
        if hasattr(layers[idx], 'LSTM_b_modal'):
            load_weights_lstm(layers,idx+gap,epoch,weights_dir,layers[idx].LSTM_b_modal,'LSTM_b_modal')
def save_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        np.save(os.path.join(weights_dir, 'mom_' + str(ind) + '_' + str(epoch)),
                vels[ind].get_value())


def load_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        vels[ind].set_value(np.load(os.path.join(
            weights_dir, 'mom_' + str(ind) + '_' + str(epoch) + '.npy')))
