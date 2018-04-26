import sys
import os
import cPickle as pickle
import cv2
import matplotlib.image as mpimg
import PIL
from PIL import Image

import copy

caffe_path ='/home/haoliangtan/Desktop/gradution_project/caffe/act-detector-scripts'
CAFFE_PYTHON_PATH = os.path.join(caffe_path, "../python")

UCFSPORT_Frame_Dir = '/home/haoliangtan/Desktop/gradution_project/caffe/dataset/UCFSports/Frames'
UCFSPORT_tubelets_Dir = '/home/haoliangtan/Desktop/my_demo/UCFSpors/tubelets'
UCFSPORT_tubes_Dir = '/home/haoliangtan/Desktop/my_demo/UCFSpors/tubes'

UCF_TubesRst = '/home/haoliangtan/Desktop/my_demo/result_tubes'
UCF_TubleRst = '/home/haoliangtan/Desktop/my_demo/result_tubelets'

sys.path.insert(0, CAFFE_PYTHON_PATH)
import caffe
from Dataset import GetDataset
from ACT_utils import *
from copy import deepcopy

K = 6
IMGSIZE = 300
MEAN = np.array([[[104, 117, 123]]], dtype=np.float32)
NFLOWS = 5
wid = 4
color_dict = {0:(255,255,0),1:(255,106,106),2:(138,43,226),3:(255,0,255),4:(0,255,0),5:(255,250,250)}
#yellow , red ,bule magenta,green snow

def add_each_tube(tube,i,v,thre = 0.1):
    if tube[1]<thre:
        return
    Result_Frame_Dir = os.path.join(UCF_TubesRst,v) #'/home/haoliangtan/Desktop/my_demo/result_tubes/001'
    Start_Frame = tube[0][0,0]
    End_Frame = tube[0][-1,0]
    for i in range(int(Start_Frame),int((End_Frame+1))):
        Sgl_Frame_Dir =os.path.join(Result_Frame_Dir,"{:0>6}".format(i)+".jpg")#'/home/haoliangtan/Desktop/my_demo/result_tubes/001/000001.jpg'
        Frame = mpimg.imread(Sgl_Frame_Dir)    #shape 404 720 3
        Frame.flags.writeable = True
        y1,x1,y2,x2 = int(tube[0][i-Start_Frame][1]),int(tube[0][i-Start_Frame][2]),int(tube[0][i-Start_Frame][3]),int(tube[0][i-Start_Frame][4])
        Frame_hole = copy.copy(Frame[x1+wid:x2-wid,y1+wid:y2-wid])
        if 1:   # the correct class  need to  implement
            Frame[x1:x2,y1:y2]=[200,200,0]  #color yellow
            Frame[x1+wid:x2-wid,y1+wid:y2-wid] = Frame_hole
            Frame = Image.fromarray(Frame)
            Frame.save(Sgl_Frame_Dir)

def add_tubes_testvideo(dname,thre = 0.3,redo=False):   #001_tubes.pkl
    # d = GetDataset(dname)
    # vlist = d.test_vlist()
    rst_tulets_dir = '/home/haoliangtan/Desktop/my_demo/result_tubes'
    end_file = os.path.join(rst_tulets_dir, "end_tubelets.pkl")

    if os.path.isfile(end_file) :     #the work has been done
        print 'the result has done ,please to cheak to view'
        print 'the current threshold  is ',thre
    else:                                        # the work need to be done
        d = GetDataset(dname)
        vlist = d.test_vlist()
        labels =d.labels
        for iv, v in enumerate(vlist):  #loop  for video order   vi: 0 v :001
            print ('Processing the {:d}/{:d}:{:s}  videos'.format(iv+1,len(vlist),v))
            tubesfile = os.path.join(UCFSPORT_tubes_Dir, v + "_tubes.pkl")    #for each short video
             #'/home/haoliangtan/Desktop/my_demo/UCFSpors/tubes/001_tubes.pkl'
            with open(tubesfile,'rb') as fid:            #default  think exist the tubes file 
                tubes = pickle.load(fid)
            for i in range(len(labels)):    #for each video and 10  labels every
                ilab_tubes = tubes[i]
                if len(ilab_tubes) == 0:
                    continue
                for tube in ilab_tubes:   #for each lable
                    add_each_tube(tube,i, v, thre=thre)
        thre = 0.3 #Done
        with open(end_file, 'wb') as fid:
            pickle.dump(thre, fid)  # OK

def add_each_tubelets(pkl,tubelsfile,nframe,thre = 0.7,v = 0):
    with open(tubelsfile, 'rb') as fid:
        tubelets,_ = pickle.load(fid)
    start_frsme = pkl+1
    End_frame = pkl+nframe
    for tubelet in tubelets:
        if tubelet[1]<thre :    #the score lower the hoped scorre
            continue
        for frame in range(start_frsme,End_frame+1):
            frame_dir = os.path.join(UCF_TubleRst,v,"{:0>6}".format(frame)+".jpg")
            step = frame-start_frsme
            y1, x1, y2, x2 = int(tubelet[step*4+2]),int(tubelet[step*4+3]),int(tubelet[step*4+4]),int(tubelet[step*4+5])
            Frame = mpimg.imread(frame_dir)  # shape 404 720 3
            Frame.flags.writeable = True
            Frame_hole = copy.copy(Frame[x1 + wid:x2 - wid, y1 + wid:y2 - wid])
            Frame[x1:x2, y1:y2] = color_dict[frame-start_frsme]
            Frame[x1 + wid:x2 - wid, y1 + wid:y2 - wid] = Frame_hole
            Frame = Image.fromarray(Frame)
            Frame.save(frame_dir)

def add_tubelest_testvideo(dname,thre= 0.1): #000001.pkl
    rst_tulets_dir = '/home/haoliangtan/Desktop/my_demo/result_tubelets'
    end_file = os.path.join(rst_tulets_dir, "end_tubelets.pkl")
    if os.path.isfile(end_file) :     #the work has been done
        print 'the result has done ,please to cheak to view'
        print 'the current threshold  is ',thre
    else:  # the work need to be done
        d = GetDataset(dname)
        vlist = d.test_vlist()
        labels = d.labels
        for iv, v in enumerate(vlist):#loop  for video order   vi: 0 v :001
            print ('Processing the {:d}/{:d}:{:s}  videos'.format(iv + 1, len(vlist), v))
            for pkl in range(d.nframes(v)-K+1):
                tubelsfile = os.path.join(UCFSPORT_tubelets_Dir,v,"{:0>6}".format(pkl+1)+".pkl")
                add_each_tubelets(pkl=pkl, tubelsfile =tubelsfile , nframe = 6,thre = thre, v = v)   #pkl:the number of frame started  tubelsfile:tubelets dir nframe:(1-6)





# add_tubes_testvideo('UCFSports',thre= 0.1)
#add_tubelest_testvideo('UCFSports',0.5)
