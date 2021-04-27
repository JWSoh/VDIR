from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--model',type=int, dest='model',default=0)
parser.add_argument('--test', dest='is_train', default=True, action='store_false')


parser.add_argument('--num',type=int, dest='num',default=0)
parser.add_argument('--sigma',type=int, dest='sigma',default=10)

parser.add_argument('--K',type=int, dest='K',default=0)


OPTIONS=parser.parse_args()

HEIGHT=96
WIDTH=96
CHANNEL=3
BATCH_SIZE=16
EPOCH=20000
LEARNING_RATE=2e-4
TF_RECORD_PATH='../TFRECORD/train_REAL_NOISE_ALL.tfrecord'
CHECK_POINT_DIR='DN'
NUM_OF_DATA=587279