import tensorflow as tf
import numpy as np
import my_txtutils

ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

author = "weights/2-layer-freeze"

input = "["

with open("Generatedprojects.txt", "w") as out:
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('weights/2-layer-freeze.meta')
        new_saver.restore(sess, author)
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)
        y = np.array([[]])

        for ch in input:
            x = my_txtutils.convert_from_alphabet(ord(ch))
            x = np.array([[x]])
            y = x
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
            out.write(ch)
        for i in range(2000000):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

            c = my_txtutils.sample_from_probabilities(yo, topn=10, temp=0.7)
            y = np.array([[c]])
            c = chr(my_txtutils.convert_to_alphabet(c))
            out.write(c)

            if c == ']':
                out.write("\n")
                h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)
            print(i)
        out.close()