import argparse
import os

from tensorflow import keras
import numpy as np

from utils import generator, model, utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',             default='0',      type=str,   help='训练使用的GPU序号，使用英文逗号,隔开，如：0,1')
parser.add_argument('--num_epoch',       default=56,       type=int,   help='训练的轮数')
parser.add_argument('--lr',              default=0.001,    type=float, help='初始学习率的大小')
parser.add_argument('--batch_size',      default=16,       type=int,   help='训练的批量大小')
parser.add_argument('--num_classes',     default=3242,     type=int,   help='分类的类别数量')
parser.add_argument('--train_list',      default='dataset/train_list.txt',  type=str, help='训练数据的数据列表路径')
parser.add_argument('--val_list',        default='dataset/test_list.txt',   type=str, help='测试数据的数据列表路径')
parser.add_argument('--resume',          default=None,     type=str,   help='预训练模型的路径，当为None则不使用预训练模型')
parser.add_argument('--model_path',      default='models', type=str,   help='模型保存的路径')
args = parser.parse_args()
utils.print_arguments(args)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Datasets
    trnlist, trnlb = utils.get_data_list(path=args.train_list)
    vallist, vallb = utils.get_data_list(path=args.val_list)
    # Generators
    trn_gen = generator.DataGenerator(list_IDs=trnlist.flatten(),
                                      labels=trnlb.flatten(),
                                      n_classes=args.num_classes,
                                      batch_size=args.batch_size)
    val_gen = generator.DataGenerator(list_IDs=vallist.flatten(),
                                      labels=vallb.flatten(),
                                      n_classes=args.num_classes,
                                      batch_size=args.batch_size)
    image_len = len(trnlist.flatten())

    # 获取模型
    mgpu = len(args.gpu.split(','))
    network = model.vggvox_resnet2d_icassp(num_classes=args.num_classes, mode='train', mgpu=mgpu)

    # 加载预训练模型
    initial_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if mgpu == 1:
                network.load_weights(os.path.join(args.resume))
            else:
                network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
            initial_epoch = int(os.path.basename(args.resume).split('-')[1])
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} audios, classes: {} '.format(args.gpu, image_len, args.num_classes))

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(args.model_path, 'resnet34-{epoch:02d}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True), normal_lr]

    network.fit_generator(generator=trn_gen,
                          steps_per_epoch=int(image_len // args.batch_size),
                          epochs=args.num_epoch,
                          initial_epoch=initial_epoch,
                          max_queue_size=10,
                          callbacks=callbacks,
                          use_multiprocessing=True,
                          validation_data=val_gen,
                          workers=6,
                          verbose=1)


# 学习率衰减
def step_decay(epoch):
    half_epoch = args.num_epoch // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.num_epoch

    milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
    gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)


if __name__ == "__main__":
    main(args)
