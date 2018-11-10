import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def u_net_bn(x, is_train=False, reuse=False, pad='SAME', n_out=3):
    """image to image translation via conditional adversarial learning"""
    _, nx, ny, nz = x.shape
    print(" * Input: size of image: (%d %d %d)" % (nx, ny, nz))
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    decay = 0.9
    gamma_init=tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope("u_net_bn", reuse=reuse):
        inputs = InputLayer(x, name='in')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv2')
        conv2 = BatchNormLayer(conv2, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv3')
        conv3 = BatchNormLayer(conv3, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv4')
        conv4 = BatchNormLayer(conv4, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv5')
        conv5 = BatchNormLayer(conv5, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv6')
        conv6 = BatchNormLayer(conv6, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=None, name='conv7')
        conv7 = BatchNormLayer(conv7, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lrelu, padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)

        up7 = DeConv2d(conv8, 512, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv7')
        up7 = BatchNormLayer(up7, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv6')
        up6 = BatchNormLayer(up6, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv5')
        up5 = BatchNormLayer(up5, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv4')
        up4 = BatchNormLayer(up4, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv3')
        up3 = BatchNormLayer(up3, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv2')
        up2 = BatchNormLayer(up2, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv1')
        up1 = BatchNormLayer(up1, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), strides=(2, 2),
                                    padding=pad, act=None, W_init=w_init, b_init=None, name='deconv0')
        up0 = BatchNormLayer(up0, decay=decay, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)

    return out

def discriminator_70x70(x1, x2, is_train=False, reuse=False):
    """image to image translation via conditional adversarial learning"""
    w_init = tf.random_normal_initializer(stddev=0.02)
    decay = 0.9
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope('discriminator_70x70', reuse=reuse):
        x = tf.concat([x1, x2], -1, name='concat')
        patch_inputs = tf.random_crop(x, [-1, 70, 70, x.shape[-1]])

        n = InputLayer(patch_inputs, name='in')
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='VALID', W_init=w_init, b_init=None, name='conv1')
        n = BatchNormLayer(n, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn1')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='VALID', W_init=w_init, b_init=None, name='conv2')
        n = BatchNormLayer(n, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn2')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='VALID', W_init=w_init, b_init=None, name='conv3')
        n = BatchNormLayer(n, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn3')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='VALID', W_init=w_init, b_init=None, name='conv4')
        n = BatchNormLayer(n, decay=decay, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='bn4')

        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init, name='out')
    return n

if __name__ == '__main__':
    print(tf.__version__)
    x = tf.placeholder("float32", [None, 256, 256, 3])
    net = u_net_bn(x)

    net = discriminator_70x70(x, x)
    print(net.outputs)

# def cyclegan_generator_resnet(image, num=9, is_train=True, reuse=False, batch_size=None, name='generator'):
#     b_init = tf.constant_initializer(value=0.0)
#     w_init = tf.truncated_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#     with tf.variable_scope(name, reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         gf_dim = 32
#
#         net_in = InputLayer(image, name='in')
#         # net_pad = PadLayer(net_in, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')
#
#         net_c1 = Conv2d(net_in, gf_dim, (7, 7), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, name='c7s1-32')  # c7s1-32   shape:(1,256,256,32)
#         net_c1 = InstanceNormLayer(net_c1, act=tf.nn.relu, name='ins1')
#
#         net_c2 = Conv2d(net_c1, gf_dim * 2, (3, 3), (2, 2), act=None,
#                 padding='SAME', W_init=w_init,name='d64')  # d64   shape:(1,128,128,64)
#         net_c2 = InstanceNormLayer(net_c2, act=tf.nn.relu, name='ins2')
#
#         net_c3 = Conv2d(net_c2, gf_dim * 4, (3, 3), (2, 2), act=None,
#                 padding='SAME', W_init=w_init, name='d128')  # d128   shape(1,64,64,128)
#         net_c3 = InstanceNormLayer(net_c3, act=tf.nn.relu, name='ins3')
#
#         n = net_c3
#         for i in range(num):
#             # n_pad = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='pad_%s' % i)
#             nn = Conv2d(n, gf_dim * 4, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=b_init, name='res/c1/%s' % i)
#             nn = InstanceNormLayer(nn, act=tf.nn.relu, name='res/ins/%s' % i)
#
#             # nn = PadLayer(nn, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='pad2_%s' % i)
#             nn = Conv2d(nn, gf_dim * 4, (3, 3), (1, 1), act=None,
#                 padding='SAME', W_init=w_init, b_init=b_init, name='res/c2/%s' % i)
#             nn = InstanceNormLayer(nn, name='res/ins2/%s' % i)
#
#             nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
#             n = nn
#
#         net_r9 = n
#         net_d1 = DeConv2d(net_r9, gf_dim * 2, (3, 3), out_size=(128,128),
#             strides=(2, 2), padding='SAME', batch_size=batch_size, act=None, name='u64')  # u64
#         net_d1 = InstanceNormLayer(net_d1, act=tf.nn.relu, name='inso1')
#
#         net_d2 = DeConv2d(net_d1, gf_dim, (3, 3), out_size=(256,256),
#             strides=(2, 2), padding='SAME',batch_size=batch_size, act=None, name='u32')  # u32
#         net_d2 = InstanceNormLayer(net_d2, act=tf.nn.relu, name='inso2')
#
#         # net_d2_pad = PadLayer(net_d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='pado')
#         net_c4 = Conv2d(net_d2, 3, (7, 7), (1, 1), act=tf.nn.tanh,
#             padding='SAME', name='c7s1-3')  # c7s1-3
#     return net_c4, net_c4.outputs
#
#
# def cyclegan_discriminator_patch(inputs, is_train=True, reuse=False, name='discriminator'):
#     df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     with tf.variable_scope(name, reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         patch_inputs = tf.random_crop(inputs, [1, 70, 70, 3])
#         net_in = InputLayer(patch_inputs, name='d/in')
#
#         # 1st
#         net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
#                         padding='SAME', W_init=w_init, name='d/h0/conv2d')  # C64
#         net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
#                         padding='SAME', W_init=w_init, name='d/h1/conv2d')
#         net_h1 = InstanceNormLayer(net_h1, act=lrelu, name="d/h1/instance_norm")
#         # 2nd
#         net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
#                         padding='SAME', W_init=w_init, name='d/h2/conv2d')
#         net_h2 = InstanceNormLayer(net_h2, act=lrelu, name="d/h2/instance_norm")
#         # 3rd
#         net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
#                         padding='SAME', W_init=w_init, name='d/h3/conv2d')
#         net_h3 = InstanceNormLayer(net_h3, act=lrelu, name="d/h3/instance_norm")
#         # output
#         net_h4 = Conv2d(net_h3, 1, (4, 4), (1, 1), act=None,
#                         padding='SAME', W_init=w_init, name='d/h4/conv2d')
#
#         logits = net_h4.outputs
#
#     return net_h4,logits
