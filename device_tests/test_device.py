import tensorflow as tf
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        a=tf.constant([1.,1.])
        b=tf.constant([2.,2.])
        c=a+b
        print(sess.run(c))

