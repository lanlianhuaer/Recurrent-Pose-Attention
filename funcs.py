import time
import os
import scipy.misc
import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import glob
def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config
def get_randNd(num_seq, flag_test=False):
    tmp_rand = np.float32(np.random.rand(4,num_seq))
    flag_mirror = [1,-1] # 1 represent not mirror ,-1 mirror
    # tmp_rand
    for ind_seq in range(num_seq):
        tmp_rand[3, ind_seq]=random.choice(flag_mirror)
    if flag_test:
        tmp_rand[:3,:]=0.5
        tmp_rand[3,:] =0
    return tmp_rand

def get_jhmdb_split(config,folder):
    saction =['brush_hair','catch','clap','climb_stairs',
   'golf','jump',
    'kick_ball','pick','pour','pullup',
    'push','run','shoot_ball',
    'shoot_bow','shoot_gun','sit',
    'stand','swing_baseball','throw',
    'walk','wave']
    # train_fnames=[[[''] for col in range(70)] for row in range(len(saction))]
    # test_fnames=[[[''] for col in range(30)] for row in range(len(saction))]
    # print 'len_saction',len(saction)
    # print(len(train_fnames),train_fnames[45][29])
    splitdir=config['jhmdb_splits'] #'/home/wbdu/Tutorials/good_practice_lstm/temporal-segment-networks/data/jhmdb_splits'
    isplit=config['split']
    # import numpy as np
    # print('dd',len(saction))
    # rgb_dir=folder_spatial
    # flow_dir=folder_temporal

    train_fnames=[]
    test_fnames=[]
    train_label=[]
    test_label=[]
    # train_length=[]
    # test_length=[]
    for iaction in range(len(saction)):
        itr = 0
        ite = 0
        fname = '%s/%s_test_split%d.txt'%(splitdir,saction[iaction],isplit)
        # print fname
        for line in open(fname):
            video_name,is_train=line.split()
            if is_train=='1':
                itr+=1
            elif is_train=='2':
                ite+=1
        train_fnames_iaction=[[''] for col in range(itr)]
        train_label_iaction=[iaction for col in range(itr)]
        # train_length_iaction=[0 for col in range(itr)]
        test_fnames_iaction=[[''] for col in range(ite)]
        test_label_iaction=[iaction for col in range(ite)]
        # test_length_iaction=[0 for col in range(ite)]
        itr = 0
        ite = 0
        for line in open(fname):
            video_name,is_train=line.split()
            assert video_name[-4:]=='.avi'
            video_name=video_name[:-4]
            pic_name=os.path.join(folder,video_name)
            # print 'pic_name',pic_name
            # video_length=len(os.listdir(pic_name))/3
            # print('video_length',video_length)
            # assert len(os.listdir(pic_name))%3==0
            if is_train=='1':
                train_fnames_iaction[itr]=pic_name
                # train_length_iaction[itr]=video_length
                # print 'pic_name',pic_name
                # train_fnames[iaction][itr]=video_name
                itr+=1
            elif is_train=='2':
                test_fnames_iaction[ite]=pic_name
                # test_length_iaction[ite]=video_length
                # test_fnames[iaction][ite]=video_name
                ite+=1
        train_fnames+=train_fnames_iaction
        train_label+=train_label_iaction
        # train_length+=train_length_iaction
        test_fnames+=test_fnames_iaction
        test_label+=test_label_iaction
        # test_length+=test_length_iaction

        # print 'train_file of saction',saction[iaction],'is ',itr,ite
    return train_fnames,train_label,test_fnames,test_label
    # return train_fnames,train_label,train_length,test_fnames,test_label,test_length
def get_jhmdb_target(config,folder):
    saction =['brush_hair','catch','clap','climb_stairs',
   'golf','jump',
    'kick_ball','pick','pour','pullup',
    'push','run','shoot_ball',
    'shoot_bow','shoot_gun','sit',
    'stand','swing_baseball','throw',
    'walk','wave']
    # train_fnames=[[[''] for col in range(70)] for row in range(len(saction))]
    # test_fnames=[[[''] for col in range(30)] for row in range(len(saction))]
    # print 'len_saction',len(saction)
    # print(len(train_fnames),train_fnames[45][29])
    splitdir=config['jhmdb_splits'] #'/home/wbdu/Tutorials/good_practice_lstm/temporal-segment-networks/data/jhmdb_splits'
    isplit=config['split']
    # import numpy as np
    # print('dd',len(saction))
    # rgb_dir=folder_spatial
    # flow_dir=folder_temporal

    train_fnames=[]
    test_fnames=[]
    train_label=[]
    test_label=[]
    # train_length=[]
    # test_length=[]
    for iaction in range(len(saction)):
        itr = 0
        ite = 0
        fname = '%s/%s_test_split%d.txt'%(splitdir,saction[iaction],isplit)
        # print fname
        for line in open(fname):
            video_name,is_train=line.split()
            if is_train=='1':
                itr+=1
            elif is_train=='2':
                ite+=1
        train_fnames_iaction=[[''] for col in range(itr)]
        train_label_iaction=[iaction for col in range(itr)]
        # train_length_iaction=[0 for col in range(itr)]
        test_fnames_iaction=[[''] for col in range(ite)]
        test_label_iaction=[iaction for col in range(ite)]
        # test_length_iaction=[0 for col in range(ite)]
        itr = 0
        ite = 0
        for line in open(fname):
            video_name,is_train=line.split()
            assert video_name[-4:]=='.avi'
            video_name=video_name[:-4]
            pic_name=os.path.join(folder,saction[iaction],video_name)
            # print pic_name
            # print 'pic_name',pic_name
            # video_length=len(os.listdir(pic_name))/3
            # print('video_length',video_length)
            # assert len(os.listdir(pic_name))%3==0
            if is_train=='1':
                train_fnames_iaction[itr]=pic_name
                # train_length_iaction[itr]=video_length
                # print 'pic_name',pic_name
                # train_fnames[iaction][itr]=video_name
                itr+=1
            elif is_train=='2':
                test_fnames_iaction[ite]=pic_name
                # test_length_iaction[ite]=video_length
                # test_fnames[iaction][ite]=video_name
                ite+=1
        train_fnames+=train_fnames_iaction
        train_label+=train_label_iaction
        # train_length+=train_length_iaction
        test_fnames+=test_fnames_iaction
        test_label+=test_label_iaction
        # test_length+=test_length_iaction
        # print 'train_file of saction',saction[iaction],'is ',itr,ite
    return train_fnames,train_label,test_fnames,test_label
    # return train_fnames,train_label,train_length,test_fnames,test_label,test_length

def unpack_configs_jhmdb(config,gpu_id='1'):


    folder_spatial=config['videos_folder_spatial']
    folder_temporal=config['videos_folder_temporal']
    folder_target =config['target_dir']
    print 'split',config['split']
    # split=config['split']
    # train_videos_spatial,val_videos_spatial,train_videos_temporal,val_videos_temporal,\
    # train_labels,val_labels\
    # train_spatial,train_label,train_spatial_length,test_spatial,test_label,test_spatial_length=get_hmdb_split(config,folder_spatial,folder_temporal)
    train_videos_spatial,train_labels,val_videos_spatial,val_labels=get_jhmdb_split(config,folder_spatial)
    train_videos_temporal,train_labels,val_videos_temporal,val_labels=get_jhmdb_split(config,folder_temporal)
    train_videos_target,train_labels,val_videos_target,val_labels= get_jhmdb_target(config,folder_target)
    # train_videos_spatial=train_fnames
    # val_videos_spatial=test_fnames
    # train_videos_temporal=train_fnames
    # val_videos_temporal=test_fnames
    # print(len(train_spatial),train_spatial[0],train_spatial_label[4])
    # np.save('/media/data/wbdu/hmdb/jhmdb_split'+str(split)+'_train_length.npy',train_length)
    # np.save('/media/data/wbdu/hmdb/jhmdb_split'+str(split)+'_test_length.npy',test_length)
    # print(tr_video_length.shape)
    # img_mean_dir = config['img_mean']
    # img_mean = sio.loadmat(img_mean_dir)
    # img_mean = img_mean['image_mean']
    if config['phase']=='train':
        if len(train_videos_temporal)%2!=0:
            train_videos_temporal.append(train_videos_temporal[-1])
            train_videos_spatial.append(train_videos_spatial[-1])
            train_videos_target.append(train_videos_target[-1])
            train_labels.append(train_labels[-1])
        if len(val_videos_temporal)%2!=0:
            val_videos_temporal.append(val_videos_temporal[-1])
            val_videos_spatial.append(val_videos_spatial[-1])
            val_videos_target.append(val_videos_target[-1])
            val_labels.append(val_labels[-1])

        if gpu_id=='1':
            train_videos_temporal=train_videos_temporal[::2]
            train_videos_spatial=train_videos_spatial[::2]
            train_videos_target=train_videos_target[::2]
            val_videos_temporal =val_videos_temporal[::2]
            val_videos_spatial =val_videos_spatial[::2]
            val_videos_target =val_videos_target[::2]
            train_labels =train_labels[::2]
            val_labels=val_labels[::2]
        else:
            train_videos_temporal=train_videos_temporal[1::2]
            train_videos_spatial=train_videos_spatial[1::2]
            train_videos_target=train_videos_target[1::2]
            val_videos_temporal =val_videos_temporal[1::2]
            val_videos_spatial =val_videos_spatial[1::2]
            val_videos_target =val_videos_target[1::2]
            train_labels =train_labels[1::2]
            val_labels=val_labels[1::2]
    # print('train_vid',train_videos[:4])
    return train_videos_spatial,val_videos_spatial,train_videos_temporal,val_videos_temporal,train_videos_target,\
               val_videos_target,\
           train_labels,val_labels

def show_pic(data):
    plt.figure()
    plt.imshow(data)
    plt.show()

def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx

def gaussian_filter_process(data,data_shape):
    data_=scipy.misc.imresize(data,data_shape,interp='bilinear')
    # data_ret=ndimage.gaussian_filter(data_, sigma=(5, 5), order=0)
    data_ret=ndimage.gaussian_filter(data_, sigma=(5, 5), order=0)
    # print attention_
    return data_ret
def get_target(config,target_jhmdb,param_rand,num_timesteps,num_joints):
    # num_joints = 15
    img_scale_x = config['img_scale_x']
    img_scale_y = config['img_scale_y']
    reg_scale_x = config['reg_scale_x']
    reg_scale_y = config['reg_scale_y']
    L=5
    L_center =3
    # reg_size = 7
    data_shape=(240,320,num_joints)
    joint_binary_mask_all=np.ones([target_jhmdb.shape[0],num_timesteps,num_joints],dtype='float32')

    video_targets=np.zeros([target_jhmdb.shape[0],num_timesteps,reg_scale_x,reg_scale_y,num_joints],dtype='float32')
    for vid_idx in range(target_jhmdb.shape[0]):
        # print('idx',vid_idx)
        target_path=target_jhmdb[vid_idx]+'/joint_positions.mat'
        tatget_vid_joint=sio.loadmat(target_path)['pos_img'] #2*15*n_frame

        joint_mask = np.zeros([240,320,num_joints,tatget_vid_joint.shape[2]],dtype='float32')
        joint_mask_binary = np.ones([tatget_vid_joint.shape[2],num_joints],dtype='float32')
        for i_frame  in range(tatget_vid_joint.shape[2]):

            for idx_joint in range(num_joints):
                pos_height = int(np.floor(tatget_vid_joint[1,idx_joint+2,i_frame]))
                pos_width =int(np.floor(tatget_vid_joint[0,idx_joint+2,i_frame]))

                if pos_height>0 and pos_height<240 and pos_width>0 and pos_width<320:
                    # print pos_height,pos_width
                    joint_mask[pos_height-5:pos_height+5,pos_width-5:pos_width+5,idx_joint,i_frame] = 1
                    joint_mask[:,:,idx_joint,i_frame] =gaussian_filter_process(joint_mask[:,:,idx_joint,i_frame],(240,320))
                else:
                    joint_mask_binary[i_frame,idx_joint]=0
                    # print 'the position of the joint is not in the img range'

        # print('target.shape',tatget_vid_joint.shape)
        # pos_x = int(param_rand[0,vid_idx])
        # pos_y = int(param_rand[1,vid_idx])
        # crop_size_x = int(param_rand[4,vid_idx])
        # crop_size_y = int(param_rand[5,vid_idx])
        flag_mirror =bool(param_rand[3,vid_idx])
        target_idx_range = range(tatget_vid_joint.shape[2])
        while len(target_idx_range)<16:
            target_idx_range.append(target_idx_range[-1])
        len_video=len(target_idx_range)
        # start_frame = int(rand_param[2,vid_idx]*(len_video-num_timesteps))
        if len_video<=32:
            interval=float(len_video-L)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')
            for i in range(num_timesteps):
                # print 'test',i,frame_range2[i]
                target_idx_range[i]=target_idx_range[frame_range2[i]+L_center]
            # print("frame_range2",frame_range2,'interval',interval)
        else:
            start_frame = int(param_rand[2,vid_idx]*(len_video-32))
            # print 'len_video',len_video,'start_frame',start_frame
            interval=float(len_video-start_frame-L)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')+start_frame
            for i in range(num_timesteps):
                # print 'test',i,frame_range2[i]
                target_idx_range[i]=target_idx_range[frame_range2[i]+L_center]

        for idx in range(num_timesteps):
            for k_joint in range(num_joints):
                target_joints=scipy.misc.imresize(joint_mask[:,:,k_joint,target_idx_range[idx]],data_shape)
                joint_binary_mask_all[vid_idx,idx,k_joint]=joint_mask_binary[target_idx_range[idx],k_joint]
                # show_pic(target_idx)
                # target_idx=target_idx[pos_x:pos_x+crop_size_x,pos_y:pos_y+crop_size_y]
                # show_pic(target_idx)
                # target_joints=scipy.misc.imresize(target_joints,(7,7,num_joints))
                target_joints = scipy.misc.imresize(target_joints,(reg_scale_x,reg_scale_y))
                # show_pic(target_idx)
                # print target_idx.shape,video_targets.shape#,target_idx

                video_targets[vid_idx,idx,:,:,k_joint]=target_joints/(np.sum(target_joints)+0.00001)
        mirror_k_joint =[ 0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11]
        if flag_mirror:
            video_targets[vid_idx,:,:,:,:]=video_targets[vid_idx,:,:,::-1,:][:,:,:,mirror_k_joint]
    video_targets =np.swapaxes(video_targets,3,2)
    joint_binary_mask_all =joint_binary_mask_all.reshape([target_jhmdb.shape[0]*num_timesteps,num_joints])
    video_targets =video_targets.reshape([target_jhmdb.shape[0]*num_timesteps,reg_scale_x*reg_scale_y,num_joints])
    video_targets_masked =video_targets*joint_binary_mask_all[:,None,:]
    dummy = np.ones([target_jhmdb.shape[0]*num_timesteps,1,num_joints],dtype='float32')
    dummy_masked=dummy*(1-joint_binary_mask_all)[:,None,:]
    video_targets_with_dummy=np.concatenate([video_targets_masked,dummy_masked],axis=1)
    return video_targets_with_dummy

def train_model_wrap(config,param_rand_epoch,train_model, shared_x, shared_mask, shared_y,
                                shared_target,rand_arr,shared_use_noise,
                     count, minibatch_index, minibatch_range, batch_size,num_seq,
                     train_filenames_epoch,
                                train_filenames_epoch_target_jhmdb,
                                 train_labels,
                     flag_para_load,mask,send_queue=None, recv_queue=None,lstm_param=True):
    num_joints = config['num_joints']
    if flag_para_load:
        msg = recv_queue.get()
        assert msg == 'copy_finished'
        if count < len(minibatch_range):
            ind_to_read = minibatch_range[count]

            name_to_read_next_temporal=train_filenames_epoch[ind_to_read,:]

            rand_param_next=param_rand_epoch[ind_to_read,:,:]#get_randNd(num_seq)
            send_queue.put(name_to_read_next_temporal)
            send_queue.put(rand_param_next)
        else:
            pass
    else:
        print('not supported')
    num_timesteps=batch_size/num_seq
    mask_pre=mask[minibatch_index,:,:].reshape(-1)
    video_targets=get_target(config,train_filenames_epoch_target_jhmdb[minibatch_index,:],
                             param_rand_epoch[minibatch_index,:,:],
                      num_timesteps,num_joints)
    shared_target.set_value(video_targets)
    shared_mask.set_value(mask_pre)
    if lstm_param:
        shared_use_noise.set_value(1.)
    else:
        shared_use_noise.set_value(0.0)
    train_label= np.zeros([num_seq,num_timesteps]).astype('int64')
    for idx in range(num_seq):
        train_label[idx,:]=train_labels[minibatch_index,idx]
    train_label = train_label.reshape(-1)
    shared_y.set_value(train_label)
    conv_fea = train_model()
    return conv_fea,mask_pre,train_label

def train_model_wrap_lstm(train_model_lstm,shared_conv,
                          shared_mask,shared_use_noise,shared_y,
                          conv_fea,mask,train_label,lstm_param=True):
    shared_conv.set_value(conv_fea)
    shared_mask.set_value(mask)
    shared_y.set_value(train_label)
    if lstm_param:
        shared_use_noise.set_value(1.)
    else:
        shared_use_noise.set_value(0.0)
    cost_ij,cost_mpii_cooking,cost_att = train_model_lstm()

    return cost_ij,cost_mpii_cooking,cost_att
def get_test_error_save(save_path,config, shared_x, shared_mask, shared_y,shared_target,shared_use_noise,
                                              shared_conv,
                       test_videos,  val_labels,
                       flag_para_load,
                       batch_size,num_seq,
                       validate_model,train_model,
                       send_queue=None, recv_queue=None,
                       flag_top_5=False,flag_batch =True):
    validation_losses = []
    validation_errors = []
    num_joints = config['num_joints']
    num_timesteps=batch_size/num_seq
    classes_num =config['classes_num']
    assert len(test_videos)==len(val_labels)

    val_labels = list(val_labels)
    test_videos=list(test_videos)
    add_idx =-1
    while len(test_videos)%num_seq!=0:
        test_videos.append(test_videos[add_idx])
        val_labels.append(val_labels[add_idx])
        add_idx=add_idx-2

    mask = np.ones([len(test_videos),num_timesteps]).astype('float32')
    mask = mask.reshape([len(test_videos)/num_seq,num_seq,num_timesteps])
    test_videos = np.asarray(test_videos).reshape([len(test_videos)/num_seq,num_seq])

    val_labels = np.asarray(val_labels).reshape([len(val_labels)/num_seq,num_seq])
    n_val_batches = test_videos.shape[0]
    print test_videos.shape
    pred = np.zeros([n_val_batches,num_seq,classes_num+1])
    shared_use_noise.set_value(0.)
    if flag_para_load:
        # send the initial message to load data, before each epoch
        vid_name_pre_temporal=test_videos[0,:]
        rand_param_test=get_randNd(num_seq,flag_test=True)
        send_queue.put(vid_name_pre_temporal)
        send_queue.put(rand_param_test)
        send_queue.put('calc_finished')
    for val_index in range(n_val_batches):
        if flag_para_load:
            # load by self or the other process
            # wait for the copying to finish
            msg = recv_queue.get()
            assert msg == 'copy_finished'
            if val_index + 1 < n_val_batches:
                name_to_read_next_temporal=test_videos[val_index + 1,:]
                send_queue.put(name_to_read_next_temporal)
                send_queue.put(rand_param_test)
        else:
            print('not supported yet')
        num_timesteps=batch_size/num_seq
        mask_pre=mask[val_index,:,:].reshape(-1)
        shared_mask.set_value(mask_pre)
        val_label= np.zeros([num_seq,num_timesteps]).astype('int64')
        for idx in range(num_seq):
            val_label[idx,:]=val_labels[val_index,idx]
        val_label = val_label.reshape(-1)
        shared_y.set_value(val_label)
        conv_fea = train_model()
        shared_conv.set_value(conv_fea)
        shared_use_noise.set_value(0.)
        loss, error,prob = validate_model()
        prob = prob.reshape([num_seq,num_timesteps,-1])
        pred[val_index,:,:classes_num]=prob[:,-1,:]
        pred[val_index,:,classes_num]=val_labels[val_index,:]

        if flag_para_load and (val_index + 1 < n_val_batches):
            send_queue.put('calc_finished')
        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)
    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)
    print this_validation_error
    np.save(save_path,pred)
    
    return this_validation_error, this_validation_loss
def get_test_error(config, shared_x, shared_mask, shared_y,shared_target,shared_use_noise,
                                              shared_conv,
                       test_file_names,  val_labels,
                       flag_para_load,
                       batch_size,num_seq,
                       validate_model,train_model,
                       send_queue=None, recv_queue=None,
                       flag_top_5=False,flag_batch =True):

    validation_losses = []
    validation_errors = []
    num_joints = config['num_joints']
    num_timesteps=batch_size/num_seq
    assert len(test_file_names)==len(val_labels)

    val_labels = list(val_labels)
    test_file_names=list(test_file_names)
    while len(test_file_names)%num_seq!=0:

        test_file_names.append(test_file_names[-1])
        val_labels.append(val_labels[-1])

    mask = np.ones([len(test_file_names),num_timesteps]).astype('float32')
    mask = mask.reshape([len(test_file_names)/num_seq,num_seq,num_timesteps])
    test_file_names = np.asarray(test_file_names).reshape([len(test_file_names)/num_seq,num_seq])

    val_labels = np.asarray(val_labels).reshape([len(val_labels)/num_seq,num_seq])
    n_val_batches = test_file_names.shape[0]
    print test_file_names.shape

    shared_use_noise.set_value(0.)
    if flag_para_load:
        # send the initial message to load data, before each epoch
        vid_name_pre_temporal=test_file_names[0,:]
        rand_param_test=get_randNd(num_seq,flag_test=True)
        send_queue.put(vid_name_pre_temporal)
        send_queue.put(rand_param_test)
        send_queue.put('calc_finished')
    for val_index in range(n_val_batches):
        if flag_para_load:
            # load by self or the other process
            # wait for the copying to finish
            msg = recv_queue.get()
            assert msg == 'copy_finished'
            if val_index + 1 < n_val_batches:
                name_to_read_next_temporal=test_file_names[val_index + 1,:]
                send_queue.put(name_to_read_next_temporal)
                send_queue.put(rand_param_test)
        else:
            print('not supported yet')
        num_timesteps=batch_size/num_seq
        mask_pre=mask[val_index,:,:].reshape(-1)
        shared_mask.set_value(mask_pre)
        val_label= np.zeros([num_seq,num_timesteps]).astype('int64')
        for idx in range(num_seq):
            val_label[idx,:]=val_labels[val_index,idx]
        val_label = val_label.reshape(-1)
        shared_y.set_value(val_label)
        conv_fea = train_model()
        shared_conv.set_value(conv_fea)
        shared_use_noise.set_value(0.)
        loss, error = validate_model()

        if flag_para_load and (val_index + 1 < n_val_batches):
            send_queue.put('calc_finished')
        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)
    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)
    return this_validation_error, this_validation_loss


def get_randNd_epoch(num_seq, n_train_batches,flag_test=False,num=-1):
    tmp_rand = np.float32(np.random.rand(n_train_batches,6,num_seq))

    crop_sizes=[256,224,192,168]
    data_shape=(256,340,3)

    pos_=[[0,0],[0,1],[1,1],[1,0],[0.5,0.5]]
    flag_mirror = [1,0] # 1 represent not mirror ,-1 mirror
    for train_batch_idx in range(n_train_batches):
        for ind_seq in range(num_seq):
            tmp_rand[train_batch_idx,:2,ind_seq]= random.choice(pos_)

            crop_size_x = random.choice(crop_sizes)
            crop_size_y = random.choice(crop_sizes)
            tmp_rand[train_batch_idx,0, ind_seq]=tmp_rand[train_batch_idx,0, ind_seq]*(data_shape[0]-crop_size_x)
            tmp_rand[train_batch_idx,1, ind_seq]=tmp_rand[train_batch_idx,1, ind_seq]*(data_shape[1]-crop_size_y)
            tmp_rand[train_batch_idx,3, ind_seq]=random.choice(flag_mirror)
            tmp_rand[train_batch_idx,4,ind_seq]=crop_size_x
            tmp_rand[train_batch_idx,5,ind_seq]=crop_size_y

    if flag_test:
        tmp_rand[:,0, :]=0.5*(data_shape[0]-224)
        tmp_rand[:,1, :]=0.5*(data_shape[1]-224)

        tmp_rand[:,2,:]=0
        tmp_rand[:,3,:] =0
        tmp_rand[:,4,:]=224
        tmp_rand[:,5,:]=224
    if num!=-1:
        tmp_rand[:,4,:]=224
        tmp_rand[:,5,:]=224
        if num==0 or num==5:
            tmp_rand[:,0,:]=0
            tmp_rand[:,1,:]=0
        elif num==1 or num==6:
            tmp_rand[:,0,:]=0
            tmp_rand[:,1,:]=116
        elif num==2 or num==7:
            tmp_rand[:,0,:]=32
            tmp_rand[:,1,:]=0
        elif num==3 or num==8:
            tmp_rand[:,0,:]=32
            tmp_rand[:,1,:]=116
        elif num==4 or num==9:
            tmp_rand[:,0,:]=16
            tmp_rand[:,1,:]=58
        else:
            print('not supported')
            print(num)
        if num>4:
            tmp_rand[:,3,:] =1
        else:
            tmp_rand[:,3,:]=0
    return tmp_rand

