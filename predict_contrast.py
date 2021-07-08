import argparse
import os
import time

import numpy as np

from utils import model, utils
from utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--audio1_path', default='audio/a_1.wav',           type=str,   help='预测第一个音频')
parser.add_argument('--audio2_path', default='audio/a_2.wav',           type=str,   help='预测第二个音频')
parser.add_argument('--threshold',   default=0.7,                       type=float, help='判断是否为同一个人的阈值')
parser.add_argument('--model_path',  default=r'models/resnet34-01.h5',  type=str,   help='模型的路径')
args = parser.parse_args()
print_arguments(args)


def main(args):
    # 获取模型
    network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1), mode='eval')

    # ==> load pre-trained model
    network_eval.load_weights(os.path.join(args.model_path), by_name=True)
    print('==> successfully loading model {}.'.format(args.model_path))

    start = time.time()
    # 获取第一个语音特征
    specs1 = utils.load_data(args.audio1_path, mode='eval')
    specs1 = np.expand_dims(np.expand_dims(specs1, 0), -1)
    feature1 = network_eval.predict(specs1)[0]

    # 获取第二个语音特征
    specs2 = utils.load_data(args.audio2_path, mode='eval')
    specs2 = np.expand_dims(np.expand_dims(specs2, 0), -1)
    feature2 = network_eval.predict(specs2)[0]
    end = time.time()

    dist = np.dot(feature1, feature2.T)
    if dist > args.threshold:
        print("%s 和 %s 为同一个人，相似度为：%f，平均预测时间：%dms" % (
            args.audio1_path, args.audio2_path, dist, round((end - start) * 1000) / 2))
    else:
        print("%s 和 %s 不是同一个人，相似度仅为：%f，平均预测时间：%dms" % (
            args.audio1_path, args.audio2_path, dist, round((end - start) * 1000) / 2))


if __name__ == "__main__":
    main(args)
