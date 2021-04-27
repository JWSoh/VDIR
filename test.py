from utils import *
import glob
import model
import imageio
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', type=str, dest='model', default='model-AWGN', choices=['model-AWGN', 'model-Real_Noise'])
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--inputpath',type=str, dest='inputpath',default='dataset')
parser.add_argument('--dataset',type=str, dest='dataset',default='Kodak24')
parser.add_argument('--sigma',type=int, dest='sigma',default=10)

parser.add_argument('--noisy', dest='noisy', default=True, action='store_false')

args=parser.parse_args()

class Test(object):
    def __init__(self, model, input_path, dataset, output_path, model_path, sigma, conf):
        self.input_path=input_path
        self.dataset=dataset
        self.output_path=output_path
        self.model_path=model_path
        self.conf=conf
        self.sigma=sigma/255.
        self.model= model

        self.save=True
        self.is_GT=args.noisy

    def __call__(self):
        img_list=np.sort(np.asarray(glob.glob('%s/%s/*.png' % (self.input_path,self.dataset))))
        print(img_list)

        input=tf.placeholder(tf.float32, shape=[None, None, None, 3])

        '''P(c|y) inference'''
        EST=model.Encoder(input, 'EST',4)

        '''Re-parametrization trick'''
        eps=tf.random_normal(tf.shape(EST.mu))
        condition= eps*tf.exp(EST.sigma / 2.) + EST.mu

        '''P(x|y,c) inference'''
        MODEL=model.Denoiser(input, condition, 'Denoise')

        output=MODEL.output

        saver=tf.train.Saver()

        count_param('Denoise')
        count_param()
        print('Sigma: ', self.sigma*255.)

        with tf.Session(config=self.conf) as sess:
            ckpt_model = os.path.join(self.model_path, self.model)
            print(ckpt_model)
            saver.restore(sess,ckpt_model)

            P = []

            if self.is_GT:
                for img_path in img_list:
                    img=imread(img_path)
                    img=img[None,:,:,:]

                    np.random.seed(0)

                    noise_img=img + np.random.standard_normal(img.shape)*self.sigma

                    out = sess.run(output, feed_dict={input:noise_img})
                    P.append(psnr(img[0]*255., np.clip(np.round(out[0]*255.), 0., 255.)))

                    if self.save:
                        if not os.path.exists('%s/Noise%d/%s'% (self.output_path, self.sigma*255., self.dataset)):
                            os.makedirs('%s/Noise%d/%s'% (self.output_path, self.sigma*255., self.dataset))
                        imageio.imsave('%s/Noise%d/%s/%s.png'% (self.output_path, self.sigma*255., self.dataset, os.path.basename(img_path[:-4])), np.uint8(np.clip(np.round(out[0] * 255.), 0., 255.)))

                print('PSNR: %.2f' % np.mean(P))

            else:
                for img_path in img_list:
                    img=imread(img_path)
                    noise_img=img[None,:,:,:]

                    out = sess.run(output, feed_dict={input:noise_img})

                    if self.save:
                        if not os.path.exists('%s/output/%s'% (self.output_path, self.dataset)):
                            os.makedirs('%s/output/%s'% (self.output_path, self.dataset))
                        imageio.imsave('%s/output/%s/%s.png'% (self.output_path, self.dataset, os.path.basename(img_path[:-4])), np.uint8(np.clip(np.round(out[0] * 255.), 0., 255.)))


if __name__== '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    conf=tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction=0.9

    Tester=Test(model = args.model,
                input_path=args.inputpath,
                dataset=args.dataset,
                output_path='results',
                model_path='Model',
                sigma=args.sigma,
                conf=conf)
    Tester()

    print('Done')