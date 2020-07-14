import argparse
import os
import time
import wave
import numpy as np
import pyaudio
import utils
import model
import tensorflow

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--n_classes', default=5994, type=int, help='class dim number')
parser.add_argument('--audio_db', default='audio_db/', type=str, help='person audio database')
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

person_feature = []
person_name = []

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


# 预测获取声纹特征
def predict(audio_path):
    specs = utils.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
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


def start_recognition():
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"

    while True:
        # 打开录音
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        i = input("按下回车键开机录音，录音%s秒中：" % RECORD_SECONDS)
        print("开始录音......")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("录音已结束!")

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # 识别对比音频库的音频
        start = time.time()
        name, p = recognition(WAVE_OUTPUT_FILENAME)
        end = time.time()
        if p > 0.8:
            print("预测时间为：%d，识别说话的为：%s，相似度为：%f" % (round((end - start) * 1000), name, p))
        else:
            print("预测时间为：%d，音频库没有该用户的语音" % round((end - start) * 1000))


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    start_recognition()
