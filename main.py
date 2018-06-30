import sys
import time
from multiprocessing import Process, Queue
import yaml
import numpy as np
import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

# sys.path.append('./lib')
from tools import (save_weights,load_weights,
                   save_momentums, load_momentums)
from funcs import ( adjust_learning_rate,
                         get_test_error,  train_model_wrap,get_randNd_epoch,
                        unpack_configs_jhmdb,proc_configs,train_model_wrap_lstm,get_test_error_save)

def train_net(config, private_config):

    # UNPACK CONFIGS
    (train_videos_spatial_jhmdb,val_videos_spatial_jhmdb,train_videos_temporal_jhmdb,val_videos_temporal_jhmdb,
     train_targets,val_targets,
           train_labels_jhmdb,val_labels_jhmdb) = unpack_configs_jhmdb(config,gpu_id=private_config['gpu_id'])
    # print('val_len',len(val_videos_spatial_jhmdb),'train_len',len(train_videos_spatial_jhmdb))
    if config['modal']=='rgb':
        train_videos = list(train_videos_spatial_jhmdb)
        test_videos = list(val_videos_spatial_jhmdb)
    else:
        train_videos = list(train_videos_temporal_jhmdb)
        test_videos = list(val_videos_temporal_jhmdb)
    print('jhmdb_len',len(train_videos),len(train_labels_jhmdb))#,len(tr_video_length_jhmdb))
    flag_para_load =config['para_load']
    gpu_send_queue = private_config['queue_gpu_send']
    gpu_recv_queue = private_config['queue_gpu_recv']

    # pycuda and zmq set up
    drv.init()
    dev = drv.Device(int(private_config['gpu'][-1]))
    ctx = dev.make_context()

    sock_gpu = zmq.Context().socket(zmq.PAIR)
    if private_config['flag_client']:
        sock_gpu.connect('tcp://localhost:{0}'.format(config['sock_gpu']))
    else:
        sock_gpu.bind('tcp://*:{0}'.format(config['sock_gpu']))

    if flag_para_load:
        sock_data_2 = zmq.Context().socket(zmq.PAIR)
        sock_data_2.connect('tcp://localhost:{0}'.format(
            private_config['sock_data_1']))

        load_send_queue = private_config['queue_t2l']
        load_recv_queue = private_config['queue_l2t']
    else:
        load_send_queue = None
        load_recv_queue = None

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_config['gpu'])
    import theano
    theano.config.on_unused_input = 'warn'

    from layers import DropoutLayer
    from inception_BN import InceptionBN, compile_models
    from attention_lstm import LSTM_softmax,compile_models_LSTM
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    ## BUILD NETWORK ##
    model = InceptionBN(config)
    model_lstm = LSTM_softmax(config)
    layers = model.layers+model_lstm.layers
    print('length_layer',len(layers))
    batch_size = model.batch_size
    num_seq =model.num_seq
    # data_batch_size=[]
    ## COMPILE FUNCTIONS ##
    ( train_model, shared_x) = compile_models(model, config)

    ( train_model_lstm, validate_model_lstm, train_error,
      learning_rate, shared_conv,shared_mask, shared_y,shared_target, rand_arr,shared_use_noise,
      vels) = compile_models_LSTM(model_lstm, config)

    total_params = model.params+model_lstm.params + vels
    # initialize gpuarrays that points to the theano shared variable
    # pass parameters and other stuff
    param_ga_list = []
    param_other_list = []
    param_ga_other_list = []
    h_list = []
    shape_list = []
    dtype_list = []
    average_fun_list = []

    for param in total_params:
        param_other = theano.shared(param.get_value())
        param_ga = \
            theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_other = \
            theano.misc.pycuda_utils.to_gpuarray(
                param_other.container.value)
        h = drv.mem_get_ipc_handle(param_ga.ptr)
        average_fun = \
            theano.function([], updates=[(param,
                                          (param + param_other) / 2.)])

        param_other_list.append(param_other)
        param_ga_list.append(param_ga)
        param_ga_other_list.append(param_ga_other)
        h_list.append(h)
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)
        average_fun_list.append(average_fun)

    # pass shape, dtype and handles
    sock_gpu.send_pyobj((shape_list, dtype_list, h_list))
    shape_other_list, dtype_other_list, h_other_list = sock_gpu.recv_pyobj()

    param_ga_remote_list = []

    # create gpuarray point to the other gpu use the passed information
    for shape_other, dtype_other, h_other in zip(shape_other_list,
                                                 dtype_other_list,
                                                 h_other_list):
        param_ga_remote = \
            gpuarray.GPUArray(shape_other, dtype_other,
                              gpudata=drv.IPCMemoryHandle(h_other))

        param_ga_remote_list.append(param_ga_remote)

    print "Information passed between 2 GPUs"

    ##########################################
    ######################### TRAIN MODEL ################################

    print '... training'

    if flag_para_load:
        gpuarray_batch_temporal = theano.misc.pycuda_utils.to_gpuarray(
            shared_x.container.value)
        h_temporal= drv.mem_get_ipc_handle(gpuarray_batch_temporal.ptr)
        sock_data_2.send_pyobj((gpuarray_batch_temporal.shape, gpuarray_batch_temporal.dtype, h_temporal))
    # gpu sync before start
    gpu_send_queue.put('before_start')
    assert gpu_recv_queue.get() == 'before_start'
    num_timesteps=batch_size/num_seq
    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []
    #################################
    num_seq=config['num_seq']
    #################################
    assert len(train_videos_spatial_jhmdb)==len(train_videos)
    while len(train_videos_spatial_jhmdb)%(num_seq)!=0:
        train_videos_spatial_jhmdb.append(train_videos_spatial_jhmdb[-1])
        train_videos.append(train_videos[-1])
        train_labels_jhmdb.append(train_labels_jhmdb[-1])
        train_targets.append(train_targets[-1])
    id_seed=train_videos_spatial_jhmdb[1]
    id_seed =int(id_seed[-1])
    max_epoch = 100

    if config['phase']=='test':
        test_epoch =100
        load_weights(layers, config['weights_dir'], test_epoch,gap=0)
        DropoutLayer.SetDropoutOff()
        save_path =config['save_path']
        this_val_error, this_val_loss = get_test_error_save(save_path,config,
             shared_x, shared_mask, shared_y,shared_target,shared_use_noise,
             shared_conv,test_videos,  val_labels_jhmdb,
            flag_para_load,
            batch_size,num_seq, validate_model_lstm,train_model,
            send_queue=load_send_queue, recv_queue=load_recv_queue)
        # report validation stats
        gpu_send_queue.put(this_val_error)
        that_val_error = gpu_recv_queue.get()
        this_val_error = (this_val_error + that_val_error) / 2.
        gpu_send_queue.put(this_val_loss)
        that_val_loss = gpu_recv_queue.get()
        this_val_loss = (this_val_loss + that_val_loss) / 2.

        if private_config['flag_verbose']:
            print('epoch %i: test loss of jhmdb %f ' %
                  (epoch, this_val_loss))
            print('epoch %i: test error of jhmdb %f %%' %
                  (epoch, this_val_error * 100.))
        val_record.append([this_val_error, this_val_loss])

        if private_config['flag_save']:
            np.save(config['weights_dir'] + 'test_record_jhmdb.npy', val_record)

        DropoutLayer.SetDropoutOn()
    else:
        while epoch < config['n_epochs']:
            seed=epoch+id_seed
            np.random.seed(seed)
            np.random.shuffle(train_videos)
            np.random.seed(seed)
            np.random.shuffle(train_labels_jhmdb)
            np.random.seed(seed)
            np.random.shuffle(train_targets)
            mask_jhmdb = np.ones([len(train_videos_spatial_jhmdb),num_timesteps]).astype('float32')
            mask_train_epoch = mask_jhmdb.reshape([len(train_videos_spatial_jhmdb)/num_seq,num_seq,num_timesteps])

            train_filenames_epoch = np.asarray(train_videos).reshape([len(train_videos)/num_seq,num_seq])
            train_labels_epoch = np.asarray(train_labels_jhmdb).reshape([len(train_labels_jhmdb)/num_seq,num_seq])
            train_targets_epoch = np.asarray(train_targets).reshape([len(train_targets)/num_seq,num_seq])

            n_train_batches = train_filenames_epoch.shape[0]
            print('n_train_batches',n_train_batches)
            minibatch_range = range(n_train_batches)
            epoch = epoch + 1
            if config['resume_train'] and epoch == 1:
                load_epoch = config['load_epoch']
                if config['modal']=='rgb':
                    load_weights(layers, config['weights_dir_load_temporal_spatial'], max_epoch,gap=0)
                else:
                    load_weights(layers, config['weights_dir_load_temporal_spatial'], max_epoch,gap=69)
                # lr_to_load = np.load(
                #    config['weights_dir_load_temporal_spatial'] + 'lr_' + str(load_epoch) + '.npy')
                # val_record = list(
                #    np.load(config['weights_dir_load_temporal_spatial'] + 'test_record_jhmdb.npy'))
                # learning_rate.set_value(lr_to_load)
                print 'learning rate: ',learning_rate.get_value()
                # load_momentums(vels, config['weights_dir_load_temporal_spatial'], max_epoch)
                epoch = load_epoch + 1
                ########################
                # ############### Test on Validation Set ##################
            param_rand_epoch=get_randNd_epoch(num_seq,n_train_batches)

            if flag_para_load:
                if config['shuffle']:
                    np.random.shuffle(minibatch_range)
                # send the initial message to load data, before each epoch
                print('minibatch_0',minibatch_range[0])
                train_file_pre_temporal=train_filenames_epoch[minibatch_range[0],:]
                param_rand = param_rand_epoch[minibatch_range[0],:,:]#get_randNd(num_seq)

                load_send_queue.put(train_file_pre_temporal)
                load_send_queue.put(param_rand)
                load_send_queue.put('calc_finished')
            count = 0
            for minibatch_index in minibatch_range:

                num_iter = (epoch - 1) * n_train_batches + count
                count = count + 1
                if count%20 == 1:
                    s = time.time()

                conv_fea,mask_pre,train_label = train_model_wrap(config,param_rand_epoch,train_model,
                                                                                   shared_x, shared_mask,
                                           shared_y, shared_target,rand_arr,shared_use_noise,
                                           count, minibatch_index,
                                           minibatch_range, batch_size,num_seq,
                                           train_filenames_epoch,
                                                      train_targets_epoch,
                                            train_labels_epoch,
                                           flag_para_load,mask_train_epoch,
                                           send_queue=load_send_queue,
                                           recv_queue=load_recv_queue)
                cost_ij,cost_nll,cost_att = train_model_wrap_lstm(train_model_lstm, shared_conv,
                                                                                         shared_mask,shared_use_noise,shared_y,
                                                                                         conv_fea,mask_pre,train_label)

                # gpu sync
                gpu_send_queue.put('after_train')
                assert gpu_recv_queue.get() == 'after_train'
                # exchanging weights
                for param_ga, param_ga_other, param_ga_remote in \
                        zip(param_ga_list, param_ga_other_list,
                            param_ga_remote_list):

                    drv.memcpy_peer(param_ga_other.ptr,
                                    param_ga_remote.ptr,
                                    param_ga_remote.dtype.itemsize *
                                    param_ga_remote.size,
                                    ctx, ctx)

                ctx.synchronize()

                # gpu sync
                gpu_send_queue.put('after_ctx_sync')
                assert gpu_recv_queue.get() == 'after_ctx_sync'

                # do average
                for average_fun in average_fun_list:
                    average_fun()
                # report train stats
                if num_iter % config['print_freq'] == 0:
                    gpu_send_queue.put(cost_ij)
                    that_cost = gpu_recv_queue.get()
                    cost_ij = (cost_ij + that_cost) / 2.

                    gpu_send_queue.put(cost_nll)
                    that_cost_nll = gpu_recv_queue.get()
                    cost_nll = (cost_nll + that_cost_nll) / 2.

                    gpu_send_queue.put(cost_att)
                    that_cost_att = gpu_recv_queue.get()
                    cost_att = (cost_att + that_cost_att) / 2.

                    if private_config['flag_verbose']:
                        print 'training @ iter = ', num_iter
                        print 'training cost:', cost_ij,'cost_nll:',cost_nll,'cost_attention:',cost_att

                    if config['print_train_error']:
                        error_ij = train_error()

                        gpu_send_queue.put(error_ij)
                        that_error = gpu_recv_queue.get()
                        error_ij = (error_ij + that_error) / 2.

                        if private_config['flag_verbose']:
                            print 'training error rate:', error_ij

                if flag_para_load and (count < len(minibatch_range)):
                    load_send_queue.put('calc_finished')

                if count%20 == 0:
                    e = time.time()
                    print "time per 20 iter:", (e - s)
            # ############### Test on Validation Set ##################
            DropoutLayer.SetDropoutOff()
            this_val_error, this_val_loss = get_test_error(config,
                 shared_x, shared_mask, shared_y,shared_target,shared_use_noise,
                 shared_conv,test_videos,  val_labels_jhmdb,
                flag_para_load,
                batch_size,num_seq, validate_model_lstm,train_model,
                send_queue=load_send_queue, recv_queue=load_recv_queue)

            # report validation stats
            gpu_send_queue.put(this_val_error)
            that_val_error = gpu_recv_queue.get()
            this_val_error = (this_val_error + that_val_error) / 2.

            gpu_send_queue.put(this_val_loss)
            that_val_loss = gpu_recv_queue.get()
            this_val_loss = (this_val_loss + that_val_loss) / 2.

            if private_config['flag_verbose']:
                print('epoch %i: test loss of jhmdb %f ' %
                      (epoch, this_val_loss))
                print('epoch %i: test error of jhmdb %f %%' %
                      (epoch, this_val_error * 100.))
            val_record.append([this_val_error, this_val_loss])
            if private_config['flag_save']:
                np.save(config['weights_dir'] + 'test_record_jhmdb.npy', val_record)

            DropoutLayer.SetDropoutOn()
            ###########################################
            # Adapt Learning Rate
            step_idx = adjust_learning_rate(config, epoch, step_idx,
                                            val_record, learning_rate)
            # Save Weights, only one of them will do
            if private_config['flag_save'] :
                if epoch % config['snapshot_freq'] == 0:
                    save_weights(layers, config['weights_dir'], epoch)
                    np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                            learning_rate.get_value())
                    save_momentums(vels, config['weights_dir'], epoch)
        print('Optimization complete.')


if __name__ == '__main__':

    with open('config_2gpu.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_2gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())

    #####################
    config['split']= 1
    config['modal']='rgb'
    config['phase']='train'
    config['classes_num']= 21
    ######################

    config['weights_dir']=config['weights_dir'][:-1]+'_'+config['modal']+'_split'+str(config['split'])+'/'
    print config['weights_dir']
    config = proc_configs(config)
    if config['modal']=='rgb':
        config['input_dim']=3
    else:
        config['input_dim']=10

    if config['phase']=='test':
        config['num_seq']=2
        config['save_path']=config['weights_dir']+'jhmdb_prob_'+config['modal']+'_'+str(config['num_timesteps'])+'.npy'
        print config['save_path']

    config['batch_size'] =config['num_timesteps']*config['num_seq']
    private_config_0 = {}
    private_config_1 = {}
    if config['split']==1:
        config['sock_gpu']=7005
        private_config_0['gpu'] = 'gpu4'
        private_config_1['gpu'] = 'gpu5'
        private_config_0['sock_data_1'] = 7006
        private_config_1['sock_data_1'] = 7007
    elif config['split']==2:
        private_config_0['gpu'] = 'gpu2'
        private_config_1['gpu'] = 'gpu3'
        config['sock_gpu']=6005
        private_config_0['sock_data_1'] = 6006
        private_config_1['sock_data_1'] = 6007
    else:
        private_config_0['gpu'] = 'gpu6'
        private_config_1['gpu'] = 'gpu7'
        config['sock_gpu']=8005
        private_config_0['sock_data_1'] = 8006
        private_config_1['sock_data_1'] = 8007
    if config['modal']=='rgb':
        config['sock_gpu']=config['sock_gpu']+700
        private_config_0['sock_data_1'] = private_config_0['sock_data_1']+700
        private_config_1['sock_data_1'] = private_config_1['sock_data_1']+700

    queue_gpu_0to1 = Queue(1)
    queue_gpu_1to0 = Queue(1)
    private_config_0['queue_gpu_send'] = queue_gpu_0to1
    private_config_0['queue_gpu_recv'] = queue_gpu_1to0
    private_config_0['gpu_id'] = '0'
    private_config_0['flag_client'] = True
    private_config_0['flag_verbose'] = True
    private_config_0['flag_save'] = True
    private_config_1['queue_gpu_send'] = queue_gpu_1to0
    private_config_1['queue_gpu_recv'] = queue_gpu_0to1

    private_config_1['gpu_id'] = '1'
    private_config_1['flag_client'] = False
    private_config_1['flag_verbose'] = False
    private_config_1['flag_save'] = False

    if config['para_load']:
        from proc_load import fun_load
        private_config_0['queue_l2t'] = Queue(1)
        private_config_0['queue_t2l'] = Queue(1)
        train_proc_0 = Process(target=train_net,
                               args=(config, private_config_0))
        load_proc_0 = Process(target=fun_load,
                              args=(dict(private_config_0.items() +
                                         config.items()),
                                    private_config_0['sock_data_1']))

        private_config_1['queue_l2t'] = Queue(1)
        private_config_1['queue_t2l'] = Queue(1)
        train_proc_1 = Process(target=train_net,
                               args=(config, private_config_1))
        load_proc_1 = Process(target=fun_load,
                              args=(dict(private_config_1.items() +
                                         config.items()),
                                    private_config_1['sock_data_1']
                                    ))

        train_proc_0.start()
        load_proc_0.start()
        train_proc_1.start()
        load_proc_1.start()

        train_proc_0.join()
        load_proc_0.join()
        train_proc_1.join()
        load_proc_1.join()

    else:
        train_proc_0 = Process(target=train_net,
                               args=(config, private_config_0))
        train_proc_1 = Process(target=train_net,
                               args=(config, private_config_1))
        train_proc_0.start()
        train_proc_1.start()
        train_proc_0.join()
        train_proc_1.join()
