import argparse
import os
import time
import uuid
import model
import keras
import tensorflow
import utils
import numpy as np
from flask import request, Flask, render_template
from flask_cors import CORS


app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
# 允许跨越访问
CORS(app)

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

# 減少显存占用并修复flask调用问题
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)
keras.backend.set_session(session)
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
    # 修复flask调用问题
    with session.as_default():
        with session.graph.as_default():
            feature = network_eval.predict(specs)[0]
    return feature


# 加载要识别的音频库
def load_audio_db(path):
    name = os.path.basename(path)[:-4]
    feature = predict(path)
    person_name.append(name)
    person_feature.append(feature)
    print("Loaded %s audio." % name)


# 声纹识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    start_time1 = time.time()
    f = request.files['audio']
    if f:
        file_path = os.path.join('audio', str(uuid.uuid1()) + "." + f.filename.split('.')[-1])
        f.save(file_path)
        name = ''
        pro = 0
        try:
            feature = predict(file_path)
            for i, person_f in enumerate(person_feature):
                # 计算相识度
                dist = np.dot(feature, person_f.T)
                if dist > pro:
                    pro = dist
                    name = person_name[i]
            result = str({"code": 0, "msg": "success", "name": name}).replace("'", '"')
            print('duration:[%.0fms]' % ((time.time() - start_time1) * 1000), result)
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


# 声纹注册接口
@app.route("/register", methods=['POST'])
def register():
    global faces_db
    f = request.files['audio']
    user_name = request.values.get("name")
    if f or user_name:
        try:
            file_path = os.path.join('audio_db', user_name + "." + f.filename.split('.')[-1])
            f.save(file_path)
            load_audio_db(file_path)
            return str({"code": 0, "msg": "success"})
        except Exception as e:
            print(e)
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 2, "msg": "audio or name is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    # 加载声纹库
    start = time.time()
    audios = os.listdir(args.audio_db)
    for audio in audios:
        path = os.path.join(args.audio_db, audio)
        load_audio_db(path)
    end = time.time()
    print('加载音频库完成，消耗时间：%fms' % (round((end - start) * 1000)))
    app.run(host='localhost', port=5000)
