import tensorflow as tf
import tensorlayer as tl
import models

batch_size = 1
lr_init = 0.0002
lr_decay = 0.5
n_epoch = 200
decay_every = 30 # 50
beta1 = 0.5
results_dir = 'results'
checkpoints_dir = 'checkpoints'
tl.files.exists_or_mkdir(results_dir)
tl.files.exists_or_mkdir(checkpoints_dir)
def train():
    """ learn to transfer A to B. """
    ###======================== DEFIINE MODEL ===============================###
    x_A = tf.placeholder("float32", [None, 256, 256, 3])
    x_B = tf.placeholder("float32", [None, 256, 256, 3])

    # Random jitter was applied by resizing the 256×256 input images to 286×286 and then randomly cropping back to size 256×256.
    x_A_ = tf.image.resize_images(x_A, size=[286, 286], method=0, align_corners=False)
    x_B_ = tf.image.resize_images(x_B, size=[286, 286], method=0, align_corners=False)
    x_A_ = tf.image.random_crop(x_A_, [256, 256], seed=0, name='A_train')
    x_B_ = tf.image.random_crop(x_B_, [256, 256], seed=0, name='B_train')

    G = models.u_net_bn(x_A_, is_train=True, reuse=False)
    fake_B = G.outputs
    D = models.discriminator_70x70(x_A_, fake_B, is_train=True, reuse=False)

    D2 = models.discriminator_70x70(x_A_, x_B_, is_train=True, reuse=True)

    d_loss1 = tl.cost.mean_squared_error(D.outputs, tf.zeros_like(D.outputs), is_mean=True) # LS-GAN
    d_loss2 = tl.cost.mean_squared_error(D2.outputs, tf.ones_like(D2.outputs), is_mean=True)
    d_loss = d_loss1 + d_loss2
    g_loss = tl.cost.mean_squared_error(D.outputs, tf.ones_like(D.outputs), is_mean=True)

    #inference
    G2 = models.u_net_bn(x_A, is_train=False, reuse=True)

    ####======================== DEFINE TRAIN OPTS ==========================###
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    g_vars = tl.layers.get_variables_with_name('u_net_bn', True, True)     # only train the decoder part of the G

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    ###============================ TRAINING ================================###
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    #TODO
    x_A_images =
    x_B_images =
    n_images_train = len(x_A_images)
    n_step_epoch = int(n_images_train / batch_size)
    ni = int(np.sqrt(batch_size))

    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr_init, decay_every, lr_decay)
            print(log)

        for step in range(n_step_epoch):
            step_time = time.time()

            #TODO
            x_A_batch =
            x_B_batch =

            _, d_loss_ = sess.run([d_optim, d_loss], feed_dict={x_A: x_A_batch, x_B: x_B_batch})
            _, g_loss_ = sess.run([g_optim, g_loss], feed_dict={x_A: x_A_batch, x_B: x_B_batch})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f"
                    % (epoch, n_epoch, step, n_step_epoch, time.time()-step_time, d_loss_, g_loss_)
        print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
        ## save model and result
        if (epoch != 0) and (epoch % 10 == 0):
            result = sess.run(G2.outputs, feed_dict={x_A: XXX})
            tl.vis.save_images(result, [ni, ni], os.path.join(results_dir, 'train_%d.png' % epoch))

            tl.files.save_npz(G.all_params, name=os.path.join(checkpoints_dir, 'g_%d.npz' % epoch, sess=sess)
            tl.files.save_npz(D.all_params, name=os.path.join(checkpoints_dir, 'd_%d.npz' % epoch, sess=sess)


if __name__ == '__main__':
    print(tf.__version__)
    print(tl.__version__)

    train()
