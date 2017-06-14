import tensorflow as tf
import numpy as np
import my_txtutils

ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

author = "checkpoints/rnn_train_1497433393-27000000"

input = "["

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1497433393-0.meta')
    new_saver.restore(sess, author)
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)
    y = np.array([[]])

    for ch in input:
        x = my_txtutils.convert_from_alphabet(ord(ch))
        x = np.array([[x]])
        y = x
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
        print(ch, end="")
    for i in range(1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        c = my_txtutils.sample_from_probabilities(yo, topn=3, temp=0.7)
        y = np.array([[c]])
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == ']':
            print("")