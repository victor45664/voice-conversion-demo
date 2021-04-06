import tensorflow as tf



def StatisticPool(x, seq_length, EPSILON=1e-5):  #x[B,T,D]
    with tf.name_scope("statistic_pool") as scope:
        seq_length = tf.cast(seq_length, dtype=tf.float32)
        mask = tf.sequence_mask(seq_length, maxlen=tf.shape(x)[1], dtype=tf.float32, name=None)  #seq_length中最长的序列要和x.shape[1]一样
        mask = tf.expand_dims(mask, axis=2)  # [N, 1, embed_depth]
        mask = tf.tile(mask, multiples=[1,1,x.shape[2]])  # [N, T_in, encoder_depth]

        x_mean, x_var = tf.nn.weighted_moments(x, axes=1,frequency_weights=mask)
        x_std = tf.sqrt(x_var + EPSILON)

        return x_mean,x_std  # [B,D]



def StatisticsReplacementLayer(x, target_mean, target_std, seq_length):

    #把输入x在时间方向上的均值和方差调整为对应的值

    x_mean,x_std=StatisticPool(x, seq_length)

    norm_x=x-tf.expand_dims(x_mean,axis=1)
    norm_x=norm_x/tf.expand_dims(x_std,axis=1)

    applied_x=norm_x*tf.expand_dims(target_std, axis=1)
    applied_x=applied_x+tf.expand_dims(target_mean,axis=1)




    return applied_x








if __name__ == '__main__':
    #Demo of usage
    import numpy as np
    x_plh = tf.placeholder(tf.float32, [2, 7, 5], 'hidden_representations')
    target_mean_plh= tf.placeholder(tf.float32, [ 2, 5 ], 'mean_plh')
    target_std_plh= tf.placeholder(tf.float32, [ 2, 5 ], 'std_plh')
    seq_length_plh = tf.placeholder(tf.int32, [2], 'seq_length')

    app_x=StatisticsReplacementLayer(x_plh, target_mean_plh, target_std_plh, seq_length_plh)

    out_mean, out_std=StatisticPool(app_x, seq_length_plh)  #to Verify the mean and std is correctly changed



    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # =====================

    x_data = np.random.random((2, 7, 5))*20
    seq_length = np.array([3, 7])
    mean=np.zeros((2,5))+5
    std=np.ones((2, 5))*1.5



    sess.run(tf.global_variables_initializer())
    out = sess.run([out_mean, out_std], feed_dict={x_plh: x_data,
                                                   seq_length_plh: seq_length,
                                                   target_mean_plh:mean,
                                                   target_std_plh:std})

    print(out)