from __future__ import absolute_import, division, print_function
from builtins import range
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import urllib2
import time

def bias_variable(shape):
    # Here we just choose to initialize our biases to 0.
    # However, this is not an agreed-upon standard and
    # some initialize the biases to 0.01 to ensure
    # that all ReLU units fire in the beginning.
    initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)

def one_hot(lst, num_elements):
    out = np.zeros((len(lst), num_elements))
    out[np.arange(len(lst)), lst] = 1
    return out

print ('Downloading Shakespeare data')
source = urllib2.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
shakespeare = source.read()
print ('Download complete')

# First we need to generate a mapping between unique
# characters 
num_chars = len(set(shakespeare))
i2c_map = {i: c for i, c in enumerate(set(shakespeare))}
c2i_map = {c: i for i, c in i2c_map.iteritems()}

tf.reset_default_graph()

num_timesteps = 30

# [num inputs per timestep, num neurons in RNN Cell, num outputs for fully connected layer]
num_neurons = 150 # [num_chars, 150, num_chars] 
batch_size  = 1

x = tf.placeholder(tf.float32, [batch_size, None, num_chars])
y = tf.placeholder(tf.float32, shape=[None, num_chars])

state = tf.placeholder(tf.float32, shape=[batch_size, num_neurons])
basic_cell = tf.contrib.rnn.GRUCell(num_units=num_neurons)
outputs, final_state = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32, initial_state=state)

# outputs :: [batch_size, timesteps, 150]
# logits  :: [batch_size, timesteps, num_chars]

w = tf.get_variable("w", shape=[num_neurons, num_chars])
b = bias_variable([num_chars])
logits = tf.tensordot(outputs, w, [[2], [0]]) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    num_epochs = 30
        
    shakespeare_trim = shakespeare * num_epochs

    num_train = len(shakespeare_trim)
    
    print("Training for %d epochs (%d characters)" % (num_epochs, num_train))
    
    current_idx = 0
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    rnn_state = tf.zeros((batch_size, num_neurons)).eval() # np.load('rnn_state.npy') # 

    # Train
    start_time = time.time()
    old_time = start_time
    # for i in range(num_epochs):
    
    chars_per_iter = batch_size * num_timesteps
    num_iterations = (num_train - 1) / chars_per_iter
    print("At %d characters per iteration, this will take %d iterations." % (chars_per_iter, num_iterations))
    for j in range(num_iterations):
        x_data = shakespeare_trim[current_idx:(current_idx + chars_per_iter)]
        y_data = shakespeare_trim[(current_idx + 1):(current_idx + chars_per_iter + 1)]

        current_idx += chars_per_iter

        x_data = [c2i_map[c] for c in x_data]
        x_batch = np.reshape(one_hot(x_data, num_chars), (batch_size, num_timesteps, num_chars))

        y_data = [c2i_map[c] for c in y_data]
        y_batch = one_hot(y_data, num_chars)

        _, rnn_state = sess.run([train_step, final_state], 
                                feed_dict={x: x_batch, y: y_batch, state: rnn_state})
        if j % 50 == 0:
            train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                            feed_dict={x: x_batch, y: y_batch, state: rnn_state})
            curr_time = time.time()
            print("iter %d / %d completed: training accuracy %g, loss %g, elapsed time %d sec, time delta %d sec"
                  %(j, num_iterations, train_accuracy, loss, (curr_time - start_time), (curr_time - old_time)))
            old_time = curr_time
    
    print("Training finished.")
    # Save the model
    save_path = saver.save(sess, "./ShakespeareRNN.ckpt")
    np.save('rnn_state', rnn_state)
    print("Model saved in file: %s" % save_path)
    
    num_chars_to_generate = 1000
    
    # seed = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
    seed = "SECOND LORD:\nNo more than a fish loves water. Is not this a strange\nfellow, my lord, that so confidently seems to undertake this\nbusiness, which he knows is not to be done; damns himself to do,\nand dares better be damn'd than to do 't.\n\n  FIRST LORD:\nYou do not know him, my lord, as we do. Certain it is\nthat he will steal himself into a man's favour, and for a week\nescape a great deal of discoveries; but when you find him out,\nyou have him ever after.\n\nBERTRAM:\nWhy, do you think he will make no deed at all of this that\nso seriously he does address himself unto?\n\nSECOND LORD:\nNone in the world; but return with an invention, and\nclap upon you two or three probable lies. But we have almost\nemboss'd him. You shall see his fall to-night; for indeed he is\nnot for your lordship's respect.\n\nFIRST LORD:\nWe'll make you some sport with the fox ere we case him.\nHe was first smok'd by the old Lord Lafeu. When his disguise and\nhe is parted, tell me what a sprat you shall find him; which you\nshall see this very night.\nSECOND LORD:\nI must go look my twigs; he shall be caught.\nBERTRAM:\nYour brother, he shall go along with me.\n\nSECOND LORD:\nAs't please your lordship. I'll leave you.\nExit\n\nBERTRAM:\nNow will I lead you to the house, and show you\nThe lass I spoke of.\n\nFIRST LORD:\nBut you say she's honest.\n\nBERTRAM:\nThat's all the fault. I spoke with her but once,\nAnd found her wondrous cold; but I sent to her,\nBy this same coxcomb that we have i' th' wind,\nTokens and letters which she did re-send;\nAnd this is all I have done. She's a fair creature;\nWill you go see her?\n\nFIRST LORD:\nWith all my heart, my lord."
    
    x_in = np.zeros( (1, len(seed), num_chars) )
    for i,c in enumerate(seed):
        x_in[0,i,:] = tf.one_hot(c2i_map[c], num_chars).eval().reshape(1,1,num_chars)
    output = ""
    
    for _ in range(num_chars_to_generate):
        rnn_output, rnn_state = sess.run([logits, final_state], feed_dict={x: x_in, state: rnn_state})
        rnn_output = rnn_output[0][0]
        next_char_idx = tf.argmax(rnn_output, axis=0).eval()
        next_char = i2c_map[next_char_idx]
        output += next_char
        x_in = tf.one_hot(next_char_idx, num_chars).eval().reshape(1,1,num_chars)
    print(output)
    
# Generation Step
# with tf.Session() as sess:
    # Restore variables from disk.
#    saver.restore(sess, "./ShakespeareRNN.ckpt")
#    print("Model restored.")
#    rnn_state = np.load('rnn_state.npy')

