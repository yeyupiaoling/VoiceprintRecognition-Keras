import os
import keras
import numpy as np
import utils
import model
import generator
import argparse
import datetime


# ===========================================
#        Parse the argument
# ===========================================
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str, help='use GPU number')
parser.add_argument('--resume', default='', type=str, help='resume model path')
parser.add_argument('--save_model_path', default='model', type=str, help='save model parent path')
parser.add_argument('--log_path', default='logs', type=str, help='save tensorboard log parent path')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_classes', default=1315, type=int, help='class dim number')
parser.add_argument('--train_list', default='dataset/train_list.txt', type=str, help='train data list path')
parser.add_argument('--val_list', default='dataset/val_list.txt', type=str, help='val data list path')
parser.add_argument('--multiprocess', default=0, type=int, help='multi process read dataset. Windows must is 0')
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
args = parser.parse_args()


def main(args):
    # ==================================
    #       Get Train/Val.
    # ==================================
    trnlist, trnlb = utils.get_voxceleb2_datalist(path=args.train_list)
    vallist, vallb = utils.get_voxceleb2_datalist(path=args.val_list)

    # construct the data generator.
    params = {'dim': (257, 250, 1),
              'mp_pooler': utils.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': args.n_classes,
              'sampling_rate': 16000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              }

    # Datasets
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    val_gen = generator.DataGenerator(partition['val'], labels['val'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)

    # ==> load pre-trained model
    mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())
    if args.resume:
        if os.path.isfile(args.resume):
            if mgpu == 1:
                network.load_weights(os.path.join(args.resume))
            else:
                network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(network.summary())
    print('==> gpu {} is, training {} images, classes: 0-{} loss: {}, aggregation: {}'
          .format(args.gpu, len(partition['train']), np.max(labels['train']), args.loss, args.aggregation_mode))

    model_path, log_path = set_path(args)
    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'weights-{epoch:02d}-{acc:.3f}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    network.fit_generator(trn_gen,
                          steps_per_epoch=int(len(partition['train']) // args.batch_size),
                          epochs=args.epochs,
                          max_queue_size=10,
                          callbacks=callbacks,
                          use_multiprocessing=True,
                          validation_data=val_gen,
                          workers=4,
                          verbose=1)


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
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


def set_path(args):
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = '{0}_{1}_{2}_{args.net}_bs{args.batch_size}_{args.optimizer}_' \
                   'lr{args.lr}_bdim{args.bottleneck_dim}'.format(date, args.aggregation_mode, args.loss, args=args)
    elif args.aggregation_mode == 'vlad':
        exp_path = '{0}_{1}_{2}_{args.net}_bs{args.batch_size}_{args.optimizer}_' \
                   'lr{args.lr}_vlad{args.vlad_cluster}_bdim{args.bottleneck_dim}'.format(date, args.aggregation_mode, args.loss, args=args)
    elif args.aggregation_mode == 'gvlad':
        exp_path = '{0}_{1}_{2}_{args.net}_bs{args.batch_size}_{args.optimizer}_lr{args.lr}_vlad{args.vlad_cluster}_' \
                   'ghost{args.ghost_cluster}_bdim{args.bottleneck_dim}'.format(date, args.aggregation_mode, args.loss,date, args=args)
    else:
        raise IOError('==> unknown aggregation mode.')
    model_path = os.path.join(args.save_model_path, exp_path)
    log_path = os.path.join(args.log_path, exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


if __name__ == "__main__":
    main(args)
