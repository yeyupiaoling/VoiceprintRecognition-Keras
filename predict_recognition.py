import argparse
import os
import shutil
import time

import numpy as np

from utils import model, utils
from utils.record import RecordAudio

parser = argparse.ArgumentParser()
parser.add_argument('--audio_db',    default='audio_db/',              type=str,   help='音频库的路径')
parser.add_argument('--threshold',   default=0.7,                      type=float, help='判断是否为同一个人的阈值')
parser.add_argument('--model_path',  default=r'models/resnet34-51.h5', type=str,   help='模型的路径')
args = parser.parse_args()

person_feature = []
person_name = []

# 获取模型
network_eval = model.vggvox_resnet2d_icassp(input_dim=(257, None, 1), mode='eval')

# 加载预训练模型
network_eval.load_weights(os.path.join(args.model_path), by_name=True)
print('==> successfully loading model {}.'.format(args.model_path))


# 预测获取声纹特征
def predict(path):
    specs = utils.load_data(path, mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    feature = network_eval.predict(specs)[0]
    return feature


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    start = time.time()
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = predict(path)
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)
    end = time.time()
    print('加载音频库完成，消耗时间：%fms' % (round((end - start) * 1000)))


# 识别声纹
def recognition(path):
    name = ''
    pro = 0
    feature = predict(path)
    for i, person_f in enumerate(person_feature):
        # 计算相识度
        dist = np.dot(feature, person_f.T)
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


# 声纹注册
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    feature = predict(save_path)
    person_name.append(user_name)
    person_feature.append(feature)


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    record_audio = RecordAudio()

    while True:
        select_fun = int(input("请选择功能，0为注册音频到声纹库，1为执行声纹识别："))
        if select_fun == 0:
            audio_path = record_audio.record()
            name = input("请输入该音频用户的名称：")
            if name == '': continue
            register(audio_path, name)
        elif select_fun == 1:
            audio_path = record_audio.record()
            name, p = recognition(audio_path)
            if p > args.threshold:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        else:
            print('请正确选择功能')
