import argparse
import os

import numpy as np
from tqdm import tqdm

from utils import model, utils
from utils.utils import print_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--list_path',   default='dataset/test_list.txt',   type=str, help='用于测试的数据列表')
parser.add_argument('--model_path',  default=r'models/resnet34-05.h5',  type=str, help='模型的路径')
args = parser.parse_args()

print_arguments(args)

# 获取模型
network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1), mode='eval')

# ==> load pre-trained model
network_eval.load_weights(os.path.join(args.model_path), by_name=True)
print('==> successfully loading model {}.'.format(args.model_path))


# 根据对角余弦值计算准确率
def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    for i in tqdm(range(0, 100)):
        threshold = i * 0.01
        y_test = (y_score >= threshold)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# 预测音频
def infer(audio_path):
    specs = utils.load_data(audio_path, mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    feature = network_eval.predict(specs)[0]
    return feature


def get_all_audio_feature(list_path):
    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    features, labels = [], []
    print('开始提取全部的音频特征...')
    for line in tqdm(lines):
        path, label = line.replace('\n', '').split('\t')
        feature = infer(path)
        features.append(feature)
        labels.append(int(label))
    return features, labels


# 计算对角余弦值
def cosin_metric(x1, x2):
    return np.dot(x1, x2.T)


def main():
    features, labels = get_all_audio_feature(args.list_path)
    scores = []
    y_true = []
    print('开始两两对比音频特征...')
    for i in tqdm(range(len(features))):
        feature_1 = features[i]
        for j in range(i, len(features)):
            feature_2 = features[j]
            score = cosin_metric(feature_1, feature_2)
            scores.append(score)
            y_true.append(int(labels[i] == labels[j]))
    accuracy, threshold = cal_accuracy(scores, y_true)
    print('当阈值为%f, 准确率最大，为：%f' % (threshold, accuracy))


if __name__ == '__main__':
    main()
