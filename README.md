# 前言

本项目说是使用Keras，但使用的都是Tensorflow下的keras接口，本项目主要是用于声纹识别，也有人称为说话人识别。本项目包括了自定义数据集的训练，声纹对比，和声纹识别。

使用环境：

 - Python 3.7
 - Tensorflow 2.3.0


# 模型下载
| 数据集 | 类别数量 | 准确率 | 下载地址 |
| :---: | :---: | :---: | :---: |
| [中文语音语料数据集](https://github.com/KuangDD/zhvoice)| 3242 | 0.999787 | [点击下载](https://download.csdn.net/download/qq_33200967/20359881) |


# 安装环境

1、安装Tensorflow，这个包含GPU功能的。
```shell script
pip install tensorflow==2.3.0 -i https://mirrors.aliyun.com/pypi/simple/
```

2、、安装其他依赖库，命令如下。
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**注意：** [libsora和pyaudio安装出错解决办法](docs/faq.md)


# 自定义数据训练

本节介绍自定义数据集训练，如何不想训练模型，可以直接看下一节，使用官方公开的模型进行声纹识别。

# 创建数据
本教程笔者使用的是[中文语音语料数据集](https://github.com/KuangDD/zhvoice) ，这个数据集一共有3242个人的语音数据，有1130000+条语音数据。如果读者有其他更好的数据集，可以混合在一起使用，但要用python的工具模块aukit处理音频，降噪和去除静音。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

在`create_data.py`写下以下代码，因为[中文语音语料数据集](https://github.com/KuangDD/zhvoice) 这个数据集是mp3格式的，作者发现这种格式读取速度很慢，所以笔者把全部的mp3格式的音频转换为wav格式，在创建数据列表之后，可能有些数据的是错误的，所以我们要检查一下，将错误的数据删除。执行下面程序完成数据准备。
```shell
python create_data.py
```

执行上面的程序之后，会生成以下的数据格式，如果要自定义数据，参考如下数据列表，前面是音频的相对路径，后面的是该音频对应的说话人的标签，就跟分类一样。
```
dataset/zhvoice/zhmagicdata/5_895/5_895_20170614203758.wav	3238
dataset/zhvoice/zhmagicdata/5_895/5_895_20170614214007.wav	3238
dataset/zhvoice/zhmagicdata/5_941/5_941_20170613151344.wav	3239
dataset/zhvoice/zhmagicdata/5_941/5_941_20170614221329.wav	3239
dataset/zhvoice/zhmagicdata/5_941/5_941_20170616153308.wav	3239
dataset/zhvoice/zhmagicdata/5_968/5_968_20170614162657.wav	3240
dataset/zhvoice/zhmagicdata/5_968/5_968_20170622194003.wav	3240
dataset/zhvoice/zhmagicdata/5_968/5_968_20170707200554.wav	3240
dataset/zhvoice/zhmagicdata/5_970/5_970_20170616000122.wav	3241
```

# 数据读取
有了上面创建的数据列表和均值标准值，就可以用于训练读取。主要是把语音数据转换短时傅里叶变换的幅度谱，使用librosa可以很方便计算音频的特征，如梅尔频谱的API为`librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用PaddlePaddle训练和预测。跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为`librosa.feature.mfcc()`。在本项目中使用的API分别是`librosa.stft()`和`librosa.magphase()`。在训练时，使用了数据增强，如随机翻转拼接，随机裁剪。经过处理，最终得到一个`257*257`的短时傅里叶变换的幅度谱。
```python
wav, sr_ret = librosa.load(audio_path, sr=sr)
linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
linear_T = linear.T
mag, _ = librosa.magphase(linear_T)
mag_T = mag.T
freq, freq_time = mag_T.shape
spec_mag = mag_T[:, :spec_len]
mean = np.mean(spec_mag, 0, keepdims=True)
std = np.std(spec_mag, 0, keepdims=True)
spec_mag = (spec_mag - mean) / (std + 1e-5)
```

# 训练

在执行训练之前，可能需要修改`train.py`中的几个参数。
 - `gpu`是指定是用那个几个GPU的，如何多卡的情况下，最好全部的GPU都使用上。
 - `resume`这个是用于恢复训练的，如何之前有训练过的模型，可以只用这个参数指定模型的路径，恢复训练。
 - `batch_size`根据自己显存的大小设置batch的大小。
 - `num_classes`是分类数量，这个可以查看上一步生成数据列表最后一个得到分类数量，但也记得加1，因为label是从0开始的。
 
最后执行`train.py`开始训练。

# 评估模型
训练结束之后会保存预测模型，我们用预测模型来预测测试集中的音频特征，然后使用音频特征进行两两对比，阈值从0到1,步长为0.01进行控制，找到最佳的阈值并计算准确率。
```shell
python eval.py
```

输出类似如下：
```
-----------  Configuration Arguments -----------
list_path: dataset/test_list.txt
model_path: models/resnet34-51.h5
------------------------------------------------
==> successfully loading model models/resnet34-51.h5.

开始提取全部的音频特征...
100%|█████████████████████████████████████████████████████| 5332/5332 [01:09<00:00, 77.06it/s]
开始两两对比音频特征...
100%|█████████████████████████████████████████████████████| 5332/5332 [01:43<00:00, 51.62it/s]
100%|█████████████████████████████████████████████████████| 100/100 [00:03<00:00, 28.04it/s]
当阈值为0.790000, 准确率最大，准确率为：0.999787
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，编写`infer()`函数，在编写模型的时候，模型是有两个输出的，第一个是模型的分类输出，第二个是音频特征输出。所以在这里要输出的是音频的特征值，有了音频的特征值就可以做声纹识别了。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值`threshold`，读者可以根据自己项目的准确度要求进行修改。
```shell
python infer_contrast.py --audio_path1=audio/a_1.wav --audio_path2=audio/b_2.wav
```

输出类似如下：
```
-----------  Configuration Arguments -----------
audio1_path: audio/b_1.wav
audio2_path: audio/a_2.wav
model_path: models/resnet34-51.h5
threshold: 0.79
------------------------------------------------
==> successfully loading model models/resnet34-51.h5.

audio/b_1.wav 和 audio/a_2.wav 不是同一个人，相似度仅为：0.453039，平均预测时间：849ms
```


# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的`infer()`预测函数，通过这两个同样获取语音的特征数据。 不同的是笔者增加了`load_audio_db()`和`register()`，以及`recognition()`，第一个函数是加载声纹库中的语音数据，这些音频就是相当于已经注册的用户，他们注册的语音数据会存放在这里，如果有用户需要通过声纹登录，就需要拿到用户的语音和语音库中的语音进行声纹对比，如果对比成功，那就相当于登录成功并且获取用户注册时的信息数据。第二个函数`register()`其实就是把录音保存在声纹库中，同时获取该音频的特征添加到待对比的数据特征中。最后`recognition()`函数中，这个函数就是将输入的语音和语音库中的语音一一对比。
有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```shell
python infer_recognition.py
```

输出类似如下：
```
-----------  Configuration Arguments -----------
audio_db: audio_db
model_path: models/resnet34-51.h5
threshold: 0.79
------------------------------------------------
==> successfully loading model models/resnet34-56.h5.

Loaded 李达康 audio.
Loaded 沙瑞金 audio.
请选择功能，0为注册音频到声纹库，1为执行声纹识别：0
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
请输入该音频用户的名称：夜雨飘零
请选择功能，0为注册音频到声纹库，1为执行声纹识别：1
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
识别说话的为：夜雨飘零，相似度为：0.920434
```

# 其他版本
 - PaddlePaddle：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - Pytorch：[VoiceprintRecognition-Pytorch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch)
 - Tensorflow：[VoiceprintRecognition-Tensorflow](https://github.com/yeyupiaoling/VoiceprintRecognition-Tensorflow)