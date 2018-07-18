import sys
import os
import pandas as pd
import numpy as np
import string
import tensorflow as tf
import math

'''
Below is a function that takes a pandas Series of text as input. Then the series is converted to list. Each item in the list
goes through is coverted to string, made lower case and stripped of surrounding empty spaces. Then entire list is converted
as numpy array to be passed back
'''
def read_x(x):
    x = np.array([list(str(line).lower().strip()) for line in x.tolist()])
    return x
'''
Next is a function to take a pandas series as an input, covert to list and return as a numpy array
'''

def read_y(y):
    return np.asarray(y.tolist())

'''
Next is function to split the data for training and validation. Validation data is helpful to see how well the model 
trained on training data generalises to unseen data. The data for validation is randomly picked by shuffling the indices
of the data. The function takes questions pairs and corresponding labels as input along with the ratio of the split.

'''
def split_train_val(x1, x2, y, ratio=0.1):
    indicies = np.arange(x1.shape[0])
    np.random.shuffle(indicies)
    num_train = int(x1.shape[0]*(1-ratio))
    train_indicies = indicies[:num_train]
    val_indicies = indicies[num_train:]

    train_x1 = x1[train_indicies, :]
    train_x2 = x2[train_indicies, :]
    train_y = y[train_indicies]

    val_x1 = x1[val_indicies, :]
    val_x2 = x2[val_indicies, :]
    val_y = y[val_indicies]

    return train_x1, train_x2, train_y, val_x1, val_x2, val_y

'''
The training and validation are picked from the shuffled indices and the data is split based on the indices. Note the 
question pairs has to be picked from indices for both training and testing.
Next, a function is created to convert the list of strings into vectors. The characters set is first 
made by concatenating the english charactersa and dictionary is created with those characters as keys while integers are the values.
Note that this character set can be inferred from the dataset to include non-english characters.
We need the maximum length of all the questions preset to quantise the vector to that size. Whenver a question is smaller in lenght when compred
to maximum length, spaces are simply appended to the text.
'''

def get_encoded_x(train_x1, train_x2, test_x1, test_x2):
    chars = string.ascii_lowercase + '? ()=+-_~"`<>,./\|[]{}!@#$%^&*:;シし' + "'"
    char_map = dict(zip(list(chars), range(len(chars))))

    max_sent_len = max([len(line) for line in np.concatenate((train_x1, train_x2, test_x1, test_x2))])
    print('max sentence length: {}'.format(max_sent_len))

    def quantize(line):
        line_padding = line + [' '] * (max_sent_len - len(line))
        encode = [char_map[char] if char in char_map.keys() else char_map[' '] for char in line_padding]
        return encode
    train_x1_encoded = np.array([quantize(line) for line in train_x1])
    train_x2_encoded = np.array([quantize(line) for line in train_x2])
    test_x1_encoded = np.array([quantize(line) for line in test_x1])
    test_x2_encoded = np.array([quantize(line) for line in test_x2])
    return train_x1_encoded, train_x2_encoded, test_x1_encoded, test_x2_encoded, max_sent_len, char_map

'''
Then there is quantisation where every character is split and encoded using the character map after padded with spaces.
Then the array of integers is converted to numpy array. Next is a fuction that combines the above functions to do 
pre-processing of the data. The data present in the form of csv is read for both traning and testing. Question 1
and question 2 are from different columns of the dataframe and split accordingly. The data is a binary whether the question 
is duplicate or not.

'''


def pre_process():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    train_x1 = read_x(train_data['question1'])
    train_x2 = read_x(train_data['question2'])
    train_y = read_y(train_data['is_duplicate'])

    test_x1 = read_x(test_data['question1'])
    test_x2 = read_x(test_data['question2'])

    train_x1, train_x2, test_x1, test_x2, max_sent_len, char_map = get_encoded_x(train_x1, train_x2, test_x1, test_x2)
    train_x1, train_x2, train_y, val_x1, val_x2, val_y = split_train_val(train_x1, train_x2, train_y)

    return train_x1, train_x2, train_y, val_x1, val_x2, val_y, test_x1, test_x2, max_sent_len, char_map

'''
Next, we build a character based CNN model. The model starts with creating embedding lookup with dimensions as 
number of characters and 50. Then, four layers are convolution are created with increasing filters and stride length.
The bigger stride looks at a long temporal dimension of the text. Each convolution layer is intiated with random variable
for weights and biases. Each layer is also followed up with a max-pooling layer. At the end, the layer is flattened.
'''

def character_CNN(tf_char_map, char_map, char_embed_dim=50):
    char_set_len = len(char_map.keys())
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

    with tf.name_scope('embedding'):
        embdded_chars = tf.nn.embedding_lookup(params=tf.Variable(tf.random_normal([char_set_len, char_embed_dim])),
                                               ids=tf_char_map,
                                               name='embedding')
        embedded_chars_expanded = tf.expand_dims(embdded_chars, -1)
    prev_layer = embedded_chars_expanded
    with tf.name_scope('Character_CNN'):
        for idx, layer in enumerate([[3, 3, 1, 16], [3, 3, 16, 32], [3, 3, 32, 64], [3, 3, 64, 128]]):
            with tf.name_scope('Conv{}'.format(idx)):
                w = tf.Variable(tf.truncated_normal(layer, stddev=1e-1), name='weights')
                b = tf.Variable(tf.truncated_normal([layer[-1]], stddev=1e-1), name='bias')
                conv = conv2d(prev_layer, w, b)
                pool = maxpool2d(conv, k=2)
                prev_layer = pool

        prev_layer = tf.reshape(prev_layer, [-1, prev_layer.shape[1]._value * prev_layer.shape[2].
                                value * prev_layer.shape[3]._value])

    return prev_layer

'''
Next a function is created two of the above described CNN is created and concatenated for each question pair. 
Then a 3 fully connected layers are created of reducing dimensions to form the final activation. The model is simialr siamese
network where two encoders are trained at the same time. 
'''

def model(x1_pls, x2_pls, char_map, keep_prob):
    out_layer1 = character_CNN(x1_pls, char_map)
    out_layer2 = character_CNN(x2_pls, char_map)
    prev = tf.concat([out_layer1, out_layer2], 1)
    with tf.name_scope('fc'):
        output_units = [1024, 512, 128, 2]
        for idx, unit in enumerate(output_units):
            if idx != 3:
                prev = tf.layers.dense(prev, units=unit, activation=tf.nn.relu)
                prev = tf.nn.dropout(prev, keep_prob)
            else:
                prev = tf.layers.dense(prev, units=unit, activation=None)
                prev = tf.nn.dropout(prev, keep_prob)
    return prev

'''
Next, a function is created to train the data. The placeholders are created for question pairs and their labels.
The output of the model created from the above function is then taken through cross entropy softmax as the loss function. 
Using the adam optimiser, the model weights are optimised. 
'''

def train(train_x1, train_x2, train_y, val_x1, val_x2, val_y, max_sent_len, char_map, epochs=2, batch_size=1024, num_classes=2):
    with tf.name_scope('Placeholders'):
        x1_pls = tf.placeholder(tf.int32, shape=[None, max_sent_len])
        x2_pls = tf.placeholder(tf.int32, shape=[None, max_sent_len])
        y_pls = tf.placeholder(tf.int64, [None])
        keep_prob = tf.placeholder(tf.float32)  # Dropout

    predict = model(x1_pls, x2_pls, char_map, keep_prob)
    with tf.name_scope('loss'):
        mean_loss = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=tf.one_hot(y_pls, num_classes))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(mean_loss)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predict, 1), y_pls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    '''
    The session is intialised for all the weights. For every epoch, the encoded data is shuffled and fed through the model.
    The same procedure is also followed the validation data.  
    '''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_indicies = np.arange(train_x1.shape[0])
        variables = [mean_loss, correct_prediction, train_step]

        iter_cnt = 0
        for e in range(epochs):
            np.random.shuffle(train_indicies)
            losses = []
            correct = 0
            for i in range(int(math.ceil(train_x1.shape[0] / batch_size))):
                start_idx = (i * batch_size) % train_x1.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                feed_dict = {x1_pls: train_x1[idx, :],
                             x2_pls: train_x2[idx, :],
                             y_pls: train_y[idx],
                             keep_prob: 0.95}
                actual_batch_size = train_y[idx].shape[0]

                loss, corr, _ = sess.run(variables, feed_dict=feed_dict)
                corr = np.array(corr).astype(np.float32)

                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)
                if iter_cnt % 10 == 0:
                    print("Minibatch {0}: with training loss = {1:.3g} and accuracy of {2:.2g}" \
                          .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                iter_cnt += 1
            total_correct = correct / train_x1.shape[0]
            total_loss = np.sum(losses) / train_x1.shape[0]
            print("Epoch {2}, Overall loss = {0:.5g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))

            if (e + 1) % 5 == 0:
                val_losses = []
                val_correct = 0
                for i in range(int(math.ceil(val_x1.shape[0] / batch_size))):
                    start_idx = (i * batch_size) % val_x1.shape[0]

                    feed_dict = {x1_pls: val_x1[start_idx:start_idx + batch_size, :],
                                 x2_pls: val_x2[start_idx:start_idx + batch_size, :],
                                 y_pls: val_y[start_idx:start_idx + batch_size],
                                 keep_prob: 1}
                    print(y_pls)
                    actual_batch_size = val_y[start_idx:start_idx + batch_size].shape[0]
                    loss, corr, _ = sess.run(variables, feed_dict=feed_dict)
                    corr = np.array(corr).astype(np.float32)
                    val_losses.append(loss * actual_batch_size)
                    val_correct += np.sum(corr)

                total_correct = val_correct / val_x1.shape[0]
                total_loss = np.sum(val_losses) / val_x1.shape[0]
                print("Validation Epoch {2}, Overall loss = {0:.5g} and accuracy of {1:.3g}" \
                      .format(total_loss, total_correct, e + 1))
            if (e+1) % 10 == 0:
                save_path = saver.save(sess, './model_{}.ckpt'.format(e))
                print("Model saved in path:{}".format(save_path))

'''
Next, a function is created to do the inference for the test data. The model is stored as a checkpoint in the above step and used here for inference

'''

def inference(test_x1, max_sent_len, batch_size=1024):
    with tf.name_scope('Placeholders'):
        x_pls1 = tf.placeholder(tf.int32, shape=[None, max_sent_len])
        keep_prob = tf.placeholder(tf.float32)  # Dropout

    predict = model(x_pls1, keep_prob)
    saver = tf.train.Saver()
    ckpt_path = tf.train.latest_checkpoint('.') 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)
        print("Model restored.")

        prediction = []
        for i in range(int(math.ceil(test_x1.shape[0] / batch_size))):
            start_idx = (i * batch_size) % test_x1.shape[0]
            prediction += sess.run([tf.argmax(predict, 1)],
                                   feed_dict={x_pls1: test_x[start_idx:start_idx + batch_size, :], keep_prob:1})[0].tolist()
        print(prediction)

'''
Next, all the functions are called in order to pre-process the data, train the model and do the inference on the test data.  
'''
train_x1, train_x2, train_y, val_x1, val_x2, val_y, test_x1, test_x2, max_sent_len, char_map = pre_process()
train(train_x1, train_x2, train_y, val_x1, val_x1, val_y, max_sent_len, char_map, 100, 1024)
inference(test_x1, test_x2, max_sent_len)
'''
Validation Epoch 25, Overall loss = 0.51399 and accuracy of 1
Epoch 26, Overall loss = 0.19037 and accuracy of 0.889
Epoch 27, Overall loss = 0.15886 and accuracy of 1
Epoch 28, Overall loss = 0.15363 and accuracy of 1
Epoch 29, Overall loss = 0.098042 and accuracy of 1
Epoch 30, Overall loss = 0.10002 and accuracy of 1
Tensor("Placeholders/Placeholder_2:0", shape=(?,), dtype=int64)
'''

