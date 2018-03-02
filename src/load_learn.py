import os
import time
import json

import numpy as np
import tensorflow as tf
from IPython import embed

from config import NetConfig, TrainConfig
from modules import *
from data_util import read_target


def main():
    net_conf = NetConfig()
    net_conf.set_conf("./net_conf.txt")
    q_units = net_conf.inference_num_units
    q_layers = net_conf.inference_num_layers
    p_units = net_conf.prediction_num_units
    p_layers = net_conf.prediction_num_layers
    latent_dim = net_conf.latent_dim
    beta = net_conf.regularize_const
    
    train_conf = TrainConfig()
    train_conf.set_conf("./train_conf.txt")
    seed = train_conf.seed
    epoch = train_conf.epoch
    batch_len = train_conf.batch_length
    batchsize = train_conf.batchsize
    data_dim = 128
    
    save_dir = train_conf.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, "model.ckpt")

    # data preparation
    train_data = read_target(train_conf.train_dir, data_dim)
    train_num = train_data.shape[1]
    train_len = train_data.shape[0]
        
    if train_conf.test:
        test_data = read_target(train_conf.test_dir)
        test_len = test_data.shape[0]
        test_num = test_data.shape[1]
    
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    x = tf.placeholder(tf.float32, [batch_len, batchsize, data_dim])
    z_mean, z_log_var = inference(x, q_units, q_layers, latent_dim)
    y = generation(x, z_mean, z_log_var, p_units, p_layers, data_dim)

    with tf.name_scope('loss'):
        recon_loss = tf.reduce_mean(-tf.reduce_sum(x*tf.log(y), axis=2))
        latent_loss = -tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        loss = recon_loss + beta * latent_loss
        
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})
        
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    
    ckpt = tf.train.get_checkpoint_state(save_dir)
    latest_model = ckpt.model_checkpoint_path
    saver.restore(sess, latest_model)

    error_log = []
    for itr in range(epoch):
        batch_idx = np.random.permutation(train_num)[:batchsize]
        start_pos = np.random.randint(train_len - batch_len)
        batch = train_data[start_pos:start_pos+batch_len, batch_idx, :]
        feed_dict = {x: batch}
        _, latent, recon, total, step = sess.run([train_step,
                                                  latent_loss,
                                                  recon_loss,
                                                  loss,
                                                  global_step],
                                                 feed_dict=feed_dict)
        error_log.append([step, latent, recon, total])
        print "step:{} latent:{}, recon:{}, total:{}".format(step, latent, recon, total)
        
        if train_conf.test and itr % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(test_num)[:batchsize]
            start_pos = np.random.randint(test_len - batch_len)
            batch = test_data[start_pos:start_pos+batch_len, batch_idx, :]
            feed_dict = {x: batch}
            latent, recon, total = sess.run([latent_loss,
                                             recon_loss,
                                             loss],
                                            feed_dict=feed_dict)
            print "-------test--------"
            print "step:{} latent:{}, recon:{}, total:{}".format(step, latent, recon, total)
            print "-------------------"
            
        if step % train_conf.log_interval == 0:
            saver.save(sess, save_file, global_step=global_step)

    error_log = np.array(error_log)
    old_error_log = np.loadtxt("error.log")
    np.savetxt("error.log", np.r_[old_error_log, error_log])

if __name__ == "__main__":
    main()
