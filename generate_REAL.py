import imageio
import os
import glob
import numpy as np
import tensorflow as tf
import threading

from time import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--labelpath', type=str, dest='labelpath', default='DIV2K_train_HR/*.png')
parser.add_argument('--datapath', type=str, dest='datapath', default='DIV2K_train_REAL_NOISE/*.png')

parser.add_argument('--labelpath2', type=str, dest='labelpath2', default='SIDD/GT/*.PNG')
parser.add_argument('--datapath2', type=str, dest='datapath2', default='SIDD/NOISY/*.PNG')
args=parser.parse_args()

labelpath=args.labelpath
datapath=args.datapath

labelpath2=args.labelpath2
datapath2=args.datapath2

tfrecord_file = "train_REAL_NOISE.tfrecord"

patches=[]
labels=[]

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def patch_generate_list(data_path,label_path,patch_h,patch_w,stride, start_num, end_num, name, grad=True):
    label_list=np.sort(np.asarray(glob.glob(label_path)))
    img_list = np.sort(np.asarray(glob.glob(data_path)))

    offset=0

    fileNum=len(label_list)

    count=0
    for n in range(start_num, end_num):
        print('%s [%d/%d]' % (name, (n+1), fileNum))
        img=imread(img_list[n])
        label=imread(label_list[n])

        x,y,ch=label.shape
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_d = img[i:i + patch_h, j:j + patch_w]
                patch_l = label[i:i + patch_h, j:j + patch_w]

                count += 1
                if grad:
                    if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -5.8:
                        patches.append(patch_d.tobytes())
                        labels.append(patch_l.tobytes())
                else:
                    patches.append(patch_d.tobytes())
                    labels.append(patch_l.tobytes())

    print('Total Patches: ', count)


def patch_to_tfrecord(tfrecord_file, labels, patches):
    np.random.seed(36)
    np.random.shuffle(labels)
    np.random.seed(36)
    np.random.shuffle(patches)
    print('Selected: ', len(labels), len(patches))

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
        if i % 10000 ==0:
            print('[%d/%d] processed' % ((i+1), len(patches)))
        write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()


def write_to_tfrecord(writer, label, binary_image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
    }))
    writer.write(example.SerializeToString())
    return

t1=time()
threads=[]

for idx in range(8):
    thread=threading.Thread(target=patch_generate_list, args=(datapath,labelpath, 256,256,180, idx*100, (idx+1)*100, 'DIV2K', True))
    threads.append(thread)

for t in threads:
    t.start()

for t in threads:
    t.join()

data1_num=len(labels)
t2=time()
print('DIV2K:', data1_num, 'Time: %.4f' % ((t2-t1)))

threads=[]

for idx in range(8):
    thread=threading.Thread(target=patch_generate_list, args=(datapath2,labelpath2, 256,256,180, idx*40, (idx+1)*40, 'SIDD', False))
    threads.append(thread)

for t in threads:
    t.start()

for t in threads:
    t.join()

t3=time()
print('SIDD:', len(labels)-data1_num, 'Time: %.4f' % ((t3-t2)))

print('*********** Patch To TFRecord ************')
patch_to_tfrecord(tfrecord_file, labels, patches)
t4=time()

print('TFRecord Time: %.4f, Overall Time: %.4f' % ((t4-t3), (t4-t1)))
print('Done')