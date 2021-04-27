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
args=parser.parse_args()

labelpath=args.labelpath
tfrecord_file = "train_DN_80000.tfrecord"

labels=[]

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

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

def patch_generate_list(label_path,patch_h,patch_w,stride, start_num, end_num, name, grad=True):
    label_list=np.sort(np.asarray(glob.glob(label_path)))

    offset=0

    fileNum=len(label_list)

    count=0
    for n in range(start_num, end_num):
        print('%s [%d/%d]' % (name, (n+1), fileNum))
        label=imread(label_list[n])

        x,y,ch=label.shape
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_l = label[i:i + patch_h, j:j + patch_w]

                count += 1
                if grad:
                    if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -5.8:
                        for m in range(8):
                            labels.append(augmentation(patch_l, m).tobytes())
                else:
                    labels.append(patch_l.tobytes())

    print('Total Patches: ', count)


def patch_to_tfrecord(tfrecord_file, labels):
    np.random.seed(36)
    np.random.shuffle(labels)
    print('Selected: ', len(labels))

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(len(labels)):
        if i % 10000 ==0:
            print('[%d/%d] processed' % ((i+1), len(labels)))
        write_to_tfrecord(writer, labels[i])
        
    writer.close()


def write_to_tfrecord(writer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))
    writer.write(example.SerializeToString())
    return

t1=time()

threads=[]

for idx in range(8):
    thread=threading.Thread(target=patch_generate_list, args=(labelpath, 256,256,120, idx*100, (idx+1)*100, 'DIV2K', True))
    threads.append(thread)

for t in threads:
    t.start()

for t in threads:
    t.join()

data1_num=len(labels)
t2=time()
print('DIV2K:', data1_num, 'Time: %.4f' % ((t2-t1)))

print('*********** Patch To TFRecord ************')
patch_to_tfrecord(tfrecord_file, labels)
t3=time()

print('TFRecord Time: %.4f, Overall Time: %.4f' % ((t3-t2), (t3-t1)))
print('Done')