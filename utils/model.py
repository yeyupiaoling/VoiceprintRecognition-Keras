from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import backbone

weight_decay = 1e-4


class VladPooling(keras.layers.Layer):
    '''
    This layer follows the NetVlad, GhostVlad
    '''

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)  # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)  # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def vggvox_resnet2d_icassp(num_classes=None, input_dim=(257, 250, 1), mode='train'):
    vlad_clusters = 8
    ghost_clusters = 2
    bottleneck_dim = 512

    inputs, x = backbone.resnet_2D_v1(input_dim=input_dim, mode=mode)
    # ===============================================
    #            Fully Connected Block 1
    # ===============================================
    x_fc = keras.layers.Conv2D(bottleneck_dim, (7, 1),
                               strides=(1, 1),
                               activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='x_fc')(x)

    # ===============================================
    #            Feature Aggregation
    # ===============================================
    x_k_center = keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1),
                                     strides=(1, 1),
                                     kernel_initializer='orthogonal',
                                     use_bias=True, trainable=True,
                                     kernel_regularizer=keras.regularizers.l2(weight_decay),
                                     bias_regularizer=keras.regularizers.l2(weight_decay),
                                     name='gvlad_center_assignment')(x)
    x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

    # ===============================================
    #            Fully Connected Block 2
    # ===============================================
    x = keras.layers.Dense(bottleneck_dim, activation='relu',
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay),
                           name='fc6')(x)

    if mode != 'eval':
        # 分类
        y = keras.layers.Dense(num_classes, activation='softmax',
                               kernel_initializer='orthogonal',
                               use_bias=False, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='prediction')(x)
    else:
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    model = keras.models.Model(inputs, y)

    if mode == 'train':
        # set up optimizer.
        opt = keras.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
