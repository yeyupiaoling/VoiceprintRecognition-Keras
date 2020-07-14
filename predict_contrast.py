import argparse
import os
import time
import numpy as np
import utils
import model
import tensorflow

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--n_classes', default=5994, type=int, help='class dim number')
parser.add_argument('--audio1_path', default='audio/a_1.wav', type=str, help='contrast person1 audio path')
parser.add_argument('--audio2_path', default='audio/b_1.wav', type=str, help='contrast person2 audio path')
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str, help='resume model path')
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
args = parser.parse_args()


def main(args):
    # 減少显存占用
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tensorflow.Session(config=config)
    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': args.n_classes,
              'sampling_rate': 16000,
              'normalize': True}

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model
    network_eval.load_weights(os.path.join(args.resume), by_name=True)
    print('==> successfully loading model {}.'.format(args.resume))

    start = time.time()
    # 获取第一个语音特征
    specs1 = utils.load_data(args.audio1_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
    specs1 = np.expand_dims(np.expand_dims(specs1, 0), -1)
    feature1 = network_eval.predict(specs1)[0]

    # 获取第二个语音特征
    specs2 = utils.load_data(args.audio2_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
    specs2 = np.expand_dims(np.expand_dims(specs2, 0), -1)
    feature2 = network_eval.predict(specs2)[0]
    end = time.time()

    dist = np.dot(feature1, feature2.T)
    if dist > 0.8:
        print("%s 和 %s 为同一个人，相似度为：%f，平均预测时间：%dms" % (
            args.audio1_path, args.audio2_path, dist, round((end - start) * 1000) / 2))
    else:
        print("%s 和 %s 不是同一个人，相似度仅为：%f，平均预测时间：%dms" % (
            args.audio1_path, args.audio2_path, dist, round((end - start) * 1000) / 2))


if __name__ == "__main__":
    main(args)
