import os
import librosa
import utils
import struct
import uuid
from tqdm import tqdm


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_wav(self, key, wav):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(wav)))
        self.data_file.write(wav)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(wav)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(wav)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


# 格式二进制转换
def convert_data(data_list_path, output_prefix, sr=16000):
    # 读取列表
    data_list = open(data_list_path, "r").readlines()
    print("train_data size:", len(data_list))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for record in tqdm(data_list):
        try:
            path, label = record.split('\t')
            key = str(uuid.uuid1())
            wav, sr_ret = librosa.load(path, sr=sr)
            assert sr_ret == sr
            # 写入对应的数据
            writer.add_wav(key, wav.tostring())
            writer.add_label('\t'.join([key, label.replace('\n', '')]))
        except Exception as e:
            print(e)


# 生成ST-CMDS数据列表
def get_st_cmds_data_list(audio_path):
    files = os.listdir(audio_path)
    sound_sum = 0
    for file in files:
        if '.wav' not in file:
            continue
        if file[:15] not in person.keys():
            person[file[:15]] = len(person)
        sound_path = os.path.join(audio_path, file)
        if sound_sum % 500 == 0:
            f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[file[:15]]))
        else:
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[file[:15]]))
        sound_sum += 1


# 生成thchs30数据列表
def get_thchs30_data_list(audio_path):
    sound_sum = 0
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            sound_path = os.path.join(root, file)
            if sound_path[-4:] != '.wav':
                continue
            if file.split('_')[0] not in person.keys():
                person[file.split('_')[0]] = len(person)
            if sound_sum % 500 == 0:
                f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[file.split('_')[0]]))
            else:
                f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[file.split('_')[0]]))
            sound_sum += 1


# 生成aishell数据列表
def get_aishell_data_list(audio_path):
    sound_sum = 0
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            sound_path = os.path.join(root, file)
            if sound_path[-4:] != '.wav':
                continue
            if os.path.dirname(sound_path).split('\\')[-1] not in person.keys():
                person[os.path.dirname(sound_path).split('\\')[-1]] = len(person)
            if sound_sum % 500 == 0:
                f_test.write(
                    '%s\t%d\n' % (sound_path.replace('\\', '/'), person[os.path.dirname(sound_path).split('\\')[-1]]))
            else:
                f_train.write(
                    '%s\t%d\n' % (sound_path.replace('\\', '/'), person[os.path.dirname(sound_path).split('\\')[-1]]))
            sound_sum += 1


# 生成VoxCeleb2数据列表
def get_vox2_data_list(train_path):
    sound_sum = 0
    person_files = os.listdir(train_path)
    for id_path in person_files:
        if id_path not in person.keys():
            person[id_path] = len(person)
        for root, dirs, files in os.walk(os.path.join(train_path, id_path)):
            for file in files:
                sound_path = os.path.join(root, file)
                if sound_sum % 500 == 0:
                    f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                else:
                    f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                sound_sum += 1


# 生成aidatatang_200zh数据列表
def get_aidatatang_200zh_data_list(train_path):
    sound_sum = 0
    person_files = os.listdir(train_path)
    for id_path in person_files:
        if id_path not in person.keys():
            person[id_path] = len(person)
        for root, dirs, files in os.walk(os.path.join(train_path, id_path)):
            for file in files:
                sound_path = os.path.join(root, file)
                if sound_sum % 500 == 0:
                    f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                else:
                    f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                sound_sum += 1


# 生成CN_Celeb数据列表
def get_CN_Celeb_200zh_data_list(train_path):
    sound_sum = 0
    person_files = os.listdir(train_path)
    for id_path in person_files:
        if id_path not in person.keys():
            person[id_path] = len(person)
        for root, dirs, files in os.walk(os.path.join(train_path, id_path)):
            for file in files:
                sound_path = os.path.join(root, file)
                if sound_sum % 500 == 0:
                    f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                else:
                    f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), person[id_path]))
                sound_sum += 1


# 删除错误或者过短的音频
def delete_error_audio(path):
    sounds_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            sound_path = os.path.join(root, file)
            if sound_path[-4:] == '.wav' or sound_path[-4:] == '.m4a':
                sounds_path.append(sound_path)
    for audio_path in tqdm(sounds_path):
        try:
            wav = utils.load_wav(audio_path, sr=16000, mode='train')
            linear_spect = utils.lin_spectogram_from_wav(wav, 160, 400, 512)
            mag, _ = librosa.magphase(linear_spect)  # magnitude
            mag_T = mag.T
            freq, time = mag_T.shape
            if time <= 250:
                os.remove(audio_path)
                print('音频过短，删除:%s' % audio_path)
        except Exception as e:
            os.remove(audio_path)
            print('音频错误，删除:%s' % audio_path)
            print("错误原因：", e)


if __name__ == '__main__':
    person = dict()
    # delete_error_audio('dataset/')
    f_train = open('dataset/train_list.txt', 'w', encoding='utf-8')
    f_test = open('dataset/test_list.txt', 'w', encoding='utf-8')
    # get_st_cmds_data_list('dataset/ST-CMDS-20170001_1-OS')
    # get_thchs30_data_list('dataset/data_thchs30')
    # get_aishell_data_list('dataset/data_aishell')
    get_vox2_data_list('dataset/dev/aac')
    get_aidatatang_200zh_data_list('dataset/aidatatang_200zh')
    get_CN_Celeb_200zh_data_list('dataset/CN-Celeb/data')
    f_test.close()
    f_train.close()
    print("总类别数量：%d" % len(person))
    print("正在把音频文件转换为一个偏移量的二进制文件...")
    convert_data('dataset/train_list.txt', 'dataset/train_data')
    convert_data('dataset/test_list.txt', 'dataset/test_data')
