from utils import *
import model
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--step', type=int, dest='step', default=0)
args=parser.parse_args()

class Train(object):
    def __init__(self, trial, step, size, batch_size, learning_rate, max_epoch, tfrecord_path, checkpoint_dir, num_of_data, conf):
        print('[*] Initialize Training')
        self.trial = trial
        self.step=step
        self.HEIGHT=size[0]
        self.WIDTH=size[1]
        self.CHANNEL=size[2]
        self.BATCH_SIZE=batch_size
        self.learning_rate=learning_rate
        self.EPOCH=max_epoch
        self.tfrecord_path=tfrecord_path
        self.checkpoint_dir=checkpoint_dir
        self.num_of_data=num_of_data
        self.conf=conf

        '''Dataset'''
        self.label = self.load_tfrecord()

        self.sigma = tf.ones(shape=[batch_size, self.HEIGHT, self.WIDTH,self.CHANNEL]) * tf.random_uniform([], 5./255., 70./255.)
        self.noise = self.sigma * tf.random_normal(shape=tf.shape(self.label))

        self.input = self.label + self.noise

        '''P(c|y) inference'''
        self.EST=model.Encoder(self.input, 'EST', feat=4)

        '''Re-parametrization trick'''
        eps=tf.random_normal(tf.shape(self.EST.mu))
        self.condition= eps*tf.exp(self.EST.sigma / 2.) + self.EST.mu

        '''P(x|y,c) inference'''
        self.MODEL=model.Denoiser(self.input, self.condition, 'Denoise')

        '''P(y|c) reconstruction'''
        self.DEC=model.Decoder(self.condition, 'DEC')

        '''DISCRIMINATOR'''
        self.DIS_real=model.Discriminator(self.input)
        self.DIS_fake=model.Discriminator(self.DEC.output, reuse=True)

        '''ESTIMATOR'''
        self.SIGMA_EST=model.Estimator(self.condition, 'EST')

    def calc_loss(self):
        self.recon=tf.losses.absolute_difference(self.label, self.MODEL.output)
        self.KL = tf.reduce_mean(0.5 * tf.reduce_mean(tf.exp(self.EST.sigma) + tf.square(self.EST.mu) - 1. - self.EST.sigma, axis=(1,2,3)))

        self.AE_recon=tf.losses.absolute_difference(self.input, self.DEC.output)

        self.EST_loss= tf.losses.absolute_difference(self.sigma, self.SIGMA_EST.output)

        '''Adversarial Loss'''
        f_logit= self.DIS_fake.logit
        r_logit=self.DIS_real.logit

        self.d_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f_logit),  logits=f_logit))
        self.d_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(r_logit), logits=r_logit))
        self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f_logit), logits=f_logit))

        self.d_loss=self.d_fake+self.d_real

        self.loss = self.recon + 1e-2 * self.KL + self.AE_recon + 1e-3 * self.g_loss + self.EST_loss

    def __call__(self):
        print('[*] Setting Train Configuration')

        self.calc_loss()

        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Denoise')
        var_EST=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='EST')

        var_DEC=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DEC')

        var_DIS=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')

        self.global_step=tf.Variable(self.step, name='global_step', trainable=False)

        '''Learning rate and the decay rules'''
        self.learning_rate=tf.train.exponential_decay(self.learning_rate,self.global_step, 100000, 0.5, staircase=True)
        self.learning_rate=tf.maximum(self.learning_rate, 2e-5)

        '''Optimizer'''
        self.opt= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=var_list+var_EST+var_DEC)

        self.d_opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=var_DIS)


        '''Summary'''
        self.summary_op=tf.summary.merge([tf.summary.scalar('loss', self.loss),
                                          tf.summary.scalar('recon', self.recon),
                                          tf.summary.scalar('KL', self.KL),
                                          tf.summary.scalar('AE_recon', self.AE_recon),
                                          tf.summary.scalar('G_loss', self.g_loss),
                                          tf.summary.scalar('D_loss',self.d_loss),
                                          tf.summary.scalar('EST_loss',self.EST_loss),
                                          tf.summary.image('1.Input',tf.clip_by_value(self.input,0.,1.),max_outputs=4),
                                          tf.summary.image('2_1.output',tf.clip_by_value(self.MODEL.output, 0., 1.),max_outputs=4),
                                          tf.summary.image('2_2.AErecon',tf.clip_by_value(self.DEC.output, 0., 1.),max_outputs=4),
                                          tf.summary.image('3.GT', self.label, max_outputs=4)
                                          ])

        '''Training'''
        for var in var_list:
            print(var.name)

        for var in var_EST:
            print(var.name)

        for var in var_DEC:
            print(var.name)

        for var in var_DIS:
            print(var.name)

        self.saver=tf.train.Saver(max_to_keep=100000)
        self.init = tf.global_variables_initializer()

        count_param(scope='Denoise')
        count_param(scope='EST')
        count_param(scope='DEC')
        count_param()

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)

            could_load, model_step=load(self.saver,sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print('==================== Load Succeeded ====================')
                assert self.step == model_step, 'The latest step and the input step do not match.'
            else:
                print('==================== No model to load ====================')


            writer=tf.summary.FileWriter('./logs%d' % self.trial, sess.graph)

            print('[*] Training Starts')

            step=self.step
            num_of_batch = self.num_of_data // self.BATCH_SIZE
            s_epoch = (step*self.BATCH_SIZE) // self.num_of_data

            epoch=s_epoch

            t2 = time.time()
            while True:
                try:
                    sess.run(self.d_opt)
                    sess.run(self.opt)
                    step += 1

                    if step % 1000 == 0:
                        t1 = t2
                        t2 = time.time()

                        loss_, recon_, KL_, AE_recon_, g_loss_, d_loss_, summary, LR_= sess.run([self.loss, self.recon, self.KL, self.AE_recon, self.g_loss, self.d_loss, self.summary_op, self.learning_rate])
                        print('Iter:', step, '%.6f =  rec/%.6f KL/%.6f AE/%.6f g/%.6f' % (loss_, recon_, KL_, AE_recon_, g_loss_), 'd_loss: %.6f' % d_loss_)
                        print('Time: %.2f' % (t2-t1), 'LR:', LR_)

                        writer.add_summary(summary, step)
                        writer.flush()

                    if step % 10000 == 0:
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step % num_of_batch == 0:
                        print('[*] Epoch:', epoch, 'Done')
                        epoch += 1

                        if epoch == self.EPOCH:
                            break

                        print('[*] Epoch:', epoch, 'Starts', 'Total iteration', step)

                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Epoch:', epoch, 'Iteration:', step)
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break

    '''Load TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['label']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return img

    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train = iterator.get_next()

        return label_train

if __name__== '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    conf=tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction=0.9

    Trainer=Train(trial = args.trial,
                  step= args.step,
                  size=[96, 96, 3],
                  batch_size=32,
                  learning_rate=2e-4,
                  max_epoch=20000,
                  tfrecord_path='../train_DN_80000.tfrecord',
                  checkpoint_dir='DN',
                  num_of_data=640000,
                  conf=conf
                  )
    Trainer()