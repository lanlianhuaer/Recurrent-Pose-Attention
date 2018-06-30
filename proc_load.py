'''
Load data in parallel with train.py
'''

import time
import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import scipy
import scipy.misc
from pylab import *
import numpy as np

import matplotlib.pyplot as plt
import glob
import os
def show_pic(data):
    plt.figure()
    plt.imshow(data)
    plt.show()

def prepare_data_rgb(video_paths,num_timesteps,num_seq,rand_param,crop_size=224,data_shape=(256,340,3),vgg_style=False):
    L=10
    # data_shape_resize=data_shape
    img_mean=np.asarray([104 ,117, 123])
    video_pics = np.zeros([num_seq,num_timesteps,data_shape[0],data_shape[1], 3],dtype='float32')
    for vid_idx in range(video_paths.shape[0]):
        video_path=video_paths[vid_idx]
        pic_names=sorted(os.listdir(video_path))

        # pos_x=int(rand_param[0,vid_idx])
        # pos_y=int(rand_param[1,vid_idx])
        # crop_size_x=int(rand_param[4,vid_idx])
        # crop_size_y=int(rand_param[5,vid_idx])

        if len(pic_names)==0:
            print video_path
        while len(pic_names)<16:
            pic_names.append(pic_names[-1])
        len_video=len(pic_names)
        if len_video<=32:
            interval=float(len_video-10)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')
            for i in range(num_timesteps):
                pic_names[i]=pic_names[frame_range2[i]+5]
        else:
            start_frame = int(rand_param[2,vid_idx]*(len_video-32))
            interval=float(len_video-start_frame-10)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')+start_frame
            for i in range(num_timesteps):
                pic_names[i]=pic_names[frame_range2[i]+5]
        #######################################################
        flag_mirror =bool(rand_param[3,vid_idx])
        for idx in range(num_timesteps):
            img_path = os.path.join(video_path,pic_names[idx])
            img=scipy.misc.imread(img_path)
            # img_mean = scipy.misc.imresize(img_mean,data_shape)

            img = scipy.misc.imresize(img, data_shape)
            img = img[:,:,::-1]
            img=img-img_mean[None,None,:]

            video_pics[vid_idx,idx,:,:,:]=img
        if flag_mirror:
            video_pics[vid_idx,:,:,:,:]=video_pics[vid_idx,:,:,::-1,:]
    video_pics=video_pics.reshape([num_seq*num_timesteps,data_shape[0],data_shape[1],3])
    video_pics =np.swapaxes(video_pics,0,3)
    return np.ascontiguousarray(video_pics, dtype='float32')#video_pics.astype('float32')
def prepare_data_flow(video_paths,num_timesteps,num_seq,rand_param,crop_size=224,data_shape=(256,340),vgg_style=False):
    num_timesteps_flow=num_timesteps
    mean_flow=128
    data_shape_resize =data_shape #(224,224)
    flow_num=10
    video_pics = np.zeros([num_seq,num_timesteps_flow,data_shape[0], data_shape[1], flow_num],dtype='float32')
    L=flow_num/2
    for vid_idx in range(video_paths.shape[0]):
        video_path=video_paths[vid_idx]
        flow_x_names=sorted(glob.glob(video_path+'/flow_x*.jpg'))
        flow_y_names=sorted(glob.glob(video_path+'/flow_y*.jpg'))
        # pos_x=int(rand_param[0,vid_idx])#*(data_shape[0]-crop_size)
        # pos_y=int(rand_param[1,vid_idx])#*(data_shape[1]-crop_size)
        # crop_size_x=int(rand_param[4,vid_idx])
        # crop_size_y=int(rand_param[5,vid_idx])
        # start_frame = int(rand_param[2,vid_idx]*(len(flow_x_names)-L-num_timesteps))

        while len(flow_x_names)<16:
            flow_x_names.append(flow_x_names[-1])
            flow_y_names.append(flow_y_names[-1])
        ##############################################
        len_video=len(flow_x_names)
        if len_video<=32:
            interval=float(len_video-L)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')
        else:
            start_frame = int(rand_param[2,vid_idx]*(len_video-32))
            interval=float(len_video-start_frame-L)/(num_timesteps-1)
            frame_range2=(interval*np.asarray(range(num_timesteps))).astype('int')+start_frame
        flow_idx=0

        for frame_idx in frame_range2:

            video_flow = np.zeros([data_shape[0], data_shape[1],flow_num])

            for ind in range(L):
                flow_x=scipy.misc.imread(flow_x_names[frame_idx+ind])
                flow_y=scipy.misc.imread(flow_y_names[frame_idx+ind])
                flow_x =scipy.misc.imresize(flow_x,data_shape_resize)
                flow_y =scipy.misc.imresize(flow_y,data_shape_resize)
                video_flow[:,:,ind*2]=flow_x
                video_flow[:,:,ind*2+1]=flow_y

            video_pics[vid_idx,flow_idx,:,:,:]=video_flow
            flow_idx=flow_idx+1

        flag_mirror =bool(rand_param[3,vid_idx])
        if flag_mirror:
            video_pics[vid_idx,:,:,:,:]=video_pics[vid_idx,:,:,::-1,:]

    video_pics=video_pics-mean_flow
    video_pics=video_pics.reshape([num_seq*num_timesteps_flow,data_shape[0], data_shape[1], flow_num])

    video_pics =np.swapaxes(video_pics,0,3)

    return np.ascontiguousarray(video_pics, dtype='float32')
def fun_load(config, sock_data_2=5001):
    send_queue = config['queue_l2t']
    recv_queue = config['queue_t2l']
    # recv_queue and send_queue are multiprocessing.Queue
    # recv_queue is only for receiving
    # send_queue is only for sending

    num_timesteps = config['num_timesteps']
    num_seq = config['num_seq']
    img_scale_x = config['img_scale_x']
    img_scale_y = config['img_scale_y']
    drv.init()
    dev = drv.Device(int(config['gpu'][-1]))
    ctx_2 = dev.make_context()

    sock_2 = zmq.Context().socket(zmq.PAIR)
    sock_2.bind('tcp://*:{0}'.format(sock_data_2))
    shape_temporal, dtype_temporal, h_temporal = sock_2.recv_pyobj()
    print 'shared_x information received',shape_temporal
    gpu_data_remote_temporal = gpuarray.GPUArray(shape_temporal, dtype_temporal,
                                        gpudata=drv.IPCMemoryHandle(h_temporal))
    gpu_data_temporal = gpuarray.GPUArray(shape_temporal, dtype_temporal)
    # print 'img_mean received'
    # The first time, do the set ups and other stuff
    # receive information for loading
    while True:
        video_name_temporal = recv_queue.get()
        rand_param = recv_queue.get()
        if config['modal']=='rgb':
            data_temporal=prepare_data_rgb(video_name_temporal,num_timesteps,num_seq,rand_param,data_shape=(img_scale_x,img_scale_y,3))
        else:
            data_temporal=prepare_data_flow(video_name_temporal,num_timesteps,num_seq,rand_param,data_shape=(img_scale_x,img_scale_y))

        gpu_data_temporal.set(data_temporal)
        # wait for computation on last minibatch to finish
        msg = recv_queue.get()
        assert msg == 'calc_finished'
        drv.memcpy_peer(gpu_data_remote_temporal.ptr,
                        gpu_data_temporal.ptr,
                        gpu_data_temporal.dtype.itemsize *
                        gpu_data_temporal.size,
                        ctx_2, ctx_2)

        ctx_2.synchronize()
        send_queue.put('copy_finished')
