import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import os
import time
import math
import numpy as np
import my_txtutils as txt
tf.set_random_seed(0)

SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001
dropout_pkeep = 0.8

datadir = "kickstarter-text/*"
codetext, valitext, bookranges = txt.read_data_files(datadir, validation=True)

epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ]

H = tf.identity(H, name='H')

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
step = 0
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[8:])

seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
sess = tf.Session()
#new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1497296355-0.meta')
#new_saver.restore(sess,'checkpoints/rnn_train_1497296355-177000000')
sess.run(tf.global_variables_initializer())
# training loop
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=30):

    feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: dropout_pkeep, lr: learning_rate, batchsize: BATCHSIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch, learning_rate)
        summary_writer.add_summary(smm, step)

    if step % _50_BATCHES == 0 and len(valitext) > 0:
        VALI_SEQLEN = 1*1024
        bsize = len(valitext) // VALI_SEQLEN
        txt.print_validation_header(len(codetext), bookranges)
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))
        vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,
                     batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        validation_writer.add_summary(smm, step)

    if step // 3 % _50_BATCHES == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("H"))]])
        rh = np.zeros([1, INTERNALSIZE * NLAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = txt.sample_from_probabilities(ryo, topn=3 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    if step // 10 % _50_BATCHES == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved file: " + saved_file)

    progress.step(reset=step % _50_BATCHES == 0)

    istate = ostate
    step += BATCHSIZE * SEQLEN
    learning_rate = 0.9995 * learning_rate