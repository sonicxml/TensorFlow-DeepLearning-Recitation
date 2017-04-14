
# coding: utf-8


# First, we do the basic setup.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# config
logs_path = "/tmp/mnist/2"

# Load mnist data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")

    # target 10 output classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


num_neurons = [784, 768, 1280, 10]

he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope('weights'):
    w1 = tf.get_variable("w1", shape=[num_neurons[0], num_neurons[1]],
                         initializer=he_init)

    w2 = tf.get_variable("w2", shape=[num_neurons[1], num_neurons[2]],
                         initializer=he_init)

    w3 = tf.get_variable("w3", shape=[num_neurons[2], num_neurons[3]],
                         initializer=he_init)

with tf.name_scope('bias-1'):
    b1 = bias_variable([num_neurons[1]])

with tf.name_scope('bias-2'):
    b2 = bias_variable([num_neurons[2]])

with tf.name_scope('bias-3'):
    b3 = bias_variable([num_neurons[3]])

keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden_layer-1'):
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    h1_drop = tf.nn.dropout(h1, keep_prob)

with tf.name_scope('hidden_layer-2'):
    h2 = tf.nn.relu(tf.matmul(h1_drop, w2) + b2)
    h2_drop = tf.nn.dropout(h2, keep_prob)

with tf.name_scope('logits'):
    y = tf.matmul(h2_drop, w3) + b3

with tf.name_scope('softmax-crossentropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

with tf.name_scope('optimize'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

summary_op = tf.summary.merge_all()

# Start an interactive session
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

for i in range(20000):
    if (i % 500 == 0):
        print 'Epoch: ', i

    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0],
                                                               y_: batch[1],
                                                               keep_prob: 0.5})
    writer.add_summary(summary, i)

# Need to change this to be clean
test_accuracy = 0
for i in range(20):
    batch = mnist.test.next_batch(500)
    test_accuracy += 500 * accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})

print("test accuracy %g" % (test_accuracy / 10000))
