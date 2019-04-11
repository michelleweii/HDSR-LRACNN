import tensorflow as tf
import pandas as pd
from skimage import io, transform
from tqdm import tqdm
import os
import numpy as np

batch_size = 128
epochs = 100
img_path = "../data/jpg"
label_file = "../data/img_label.csv"
img_df = pd.read_csv(label_file)
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

def load_data():
    #use oxford flower17 dataset
    X_imgs = []
    y_label = []
    for i in tqdm(range(len(img_df))):
        img = transform.resize(image=io.imread(os.path.join(img_path, img_df.iloc[i, 0])), output_shape=(224,224))
        X_imgs.append(img)
        y_label.append(img_df.iloc[i,1]-1)
    X_imgs = np.array(X_imgs)
    y_label = tf.one_hot(y_label, depth=17)
    return X_imgs, y_label

def vgg16(x):
    with tf.variable_scope("block1"):
        with tf.variable_scope("block1_conv1"):
            weight1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.1), name="weight1")
            biases1 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[64]), name="biases1")
            c1 = tf.nn.bias_add(tf.nn.conv2d(x, filter=weight1, strides=[1, 1, 1, 1], padding="SAME"),
                                biases1) # 左右各padding_1
            r1 = tf.nn.relu(c1)

        with tf.variable_scope("block1_conv2"):
            weight2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name="weight2")
            biases2 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[64]), name="biases2")
            c2 = tf.nn.bias_add(tf.nn.conv2d(r1, filter=weight2, strides=[1, 1, 1, 1], padding="SAME"),
                                biases2) # 左右各padding_1
            r2 = tf.nn.relu(c2)

        with tf.variable_scope("block1_pool"):
            m1 = tf.nn.max_pool(r2, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="VALID", name="pool_1")

    with tf.variable_scope("block2"):
        with tf.variable_scope("block2_conv1"):
            weight3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=0.1), name="weight3")
            biases3 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[128]), name="biases3")
            c3 = tf.nn.bias_add(tf.nn.conv2d(m1, filter=weight3, strides=[1, 1, 1, 1], padding="SAME"), biases3)
            r3 = tf.nn.relu(c3)

        with tf.variable_scope("block2_conv2"):
            weight4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=0.1), name="weight4")
            biases4 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[64]), name="biases4")
            c4 = tf.nn.bias_add(tf.nn.conv2d(r3, filter=weight4, strides=[1, 1, 1, 1], padding="SAME"), biases4)
            r4 = tf.nn.relu(c4)

        with tf.variable_scope("block2_pool"):
            m2 = tf.nn.max_pool(r4, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="VALID", name="pooling_2")

    with tf.variable_scope("block3"):
        with tf.variable_scope("block3_conv1"):
            weight5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=0.1), name="weight5")
            biases5 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[256]), name="biases5")
            c5 = tf.nn.bias_add(tf.nn.conv2d(m2, filter=weight5, strides=[1, 1, 1, 1], padding="SAME"), biases5)
            r5 = tf.nn.relu(c5)

        with tf.variable_scope("block3_conv2"):
            weight6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=0.1), name="weight6")
            biases6 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[256]), name="biases6")
            c6 = tf.nn.bias_add(tf.nn.conv2d(r5, filter=weight6, strides=[1, 1, 1, 1], padding="SAME"), biases6)
            r6 = tf.nn.relu(c6)

        with tf.variable_scope("block3_conv3"):
            weight7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=0.1), name="weight7")
            biases7 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[256]), name="biases7")
            c7 = tf.nn.bias_add(tf.nn.conv2d(r6, filter=weight7, strides=[1, 1, 1, 1], padding="SAME"), biases7)
            r7 = tf.nn.relu(c7)

        with tf.variable_scope("block3_pool"):
            m3 = tf.nn.max_pool(r7, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="VALID", name="pooling_3")

    with tf.variable_scope("block4"):
        with tf.variable_scope("block4_conv1"):
            weight8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=0.1), name="weight8")
            biases8 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases8")
            c8 = tf.nn.bias_add(tf.nn.conv2d(m3, filter=weight8, strides=[1, 1, 1, 1], padding="SAME"), biases8)
            r8 = tf.nn.relu(c8)

        with tf.variable_scope("block4_conv2"):
            weight9 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=0.1), name="weight9")
            biases9 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases9")
            c9 = tf.nn.bias_add(tf.nn.conv2d(r8, filter=weight9, strides=[1, 1, 1, 1], padding="SAME"), biases9)
            r9 = tf.nn.relu(c9)

        with tf.variable_scope("block3_conv3"):
            weight10 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=0.1), name="weight10")
            biases10 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases10")
            c10 = tf.nn.bias_add(tf.nn.conv2d(r9, filter=weight10, strides=[1, 1, 1, 1], padding="SAME"), biases10)
            r10 = tf.nn.relu(c10)

        with tf.variable_scope("block4_pool"):
            m4 = tf.nn.max_pool(r10, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="VALID", name="pooling_4")

    with tf.variable_scope("block5"):
        with tf.variable_scope("block5_conv1"):
            weight11 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=0.1), name="weight11")
            biases11 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases11")
            c11 = tf.nn.bias_add(tf.nn.conv2d(m4, filter=weight11, strides=[1, 1, 1, 1], padding="SAME"), biases11)
            r11 = tf.nn.relu(c11)

        with tf.variable_scope("block5_conv2"):
            weight12 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=0.1), name="weight12")
            biases12 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases12")
            c12 = tf.nn.bias_add(tf.nn.conv2d(r11, filter=weight12, strides=[1, 1, 1, 1], padding="SAME"), biases12)
            r12 = tf.nn.relu(c12)

        with tf.variable_scope("block5_conv3"):
            weight13 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=0.1), name="weight13")
            biases13 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[512]), name="biases13")
            c13 = tf.nn.bias_add(tf.nn.conv2d(r12, filter=weight13, strides=[1, 1, 1, 1], padding="SAME"), biases13)
            r13 = tf.nn.relu(c13)

        with tf.variable_scope("block5_pool"):
            m5 = tf.nn.max_pool(r13, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="VALID", name="pooling_5")

    with tf.variable_scope("flatten"):
        m5_shape_li = m5.get_shape().as_list()
        f5 = tf.reshape(tensor=m5, shape=[-1, m5_shape_li[1] * m5_shape_li[2] * m5_shape_li[3]], name="flatten")

    with tf.variable_scope("fc1"):
        weight14 = tf.Variable(tf.truncated_normal(shape=[m5_shape_li[1] * m5_shape_li[2] * m5_shape_li[3], 4096],
                                                   stddev=0.1), name="weight14")
        biases14 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[4096]), name="biases14")
        o14 = tf.matmul(f5, weight14) + biases14
        r14 = tf.nn.relu(o14)
        d14 = tf.nn.dropout(r14, keep_prob=0.5)

    with tf.variable_scope("fc2"):
        weight15 = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=0.1), name="weight15")
        biases15 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[4096]), name="biases15")
        o15 = tf.matmul(d14, weight15) + biases15
        r15 = tf.nn.relu(o15)
        d15 = tf.nn.dropout(r15, keep_prob=0.5)

    with tf.variable_scope("predictions"):
        weight16 = tf.Variable(tf.truncated_normal(shape=[4096, 1000]), name="weight16")
        biases16 = tf.Variable(tf.constant(0, shape=[1000], dtype=tf.float32), name="biases16")

        o16 = tf.matmul(d15, weight16) + biases16
        output = tf.nn.relu(o16)

    return output

if __name__ == "__main__":
    with tf.variable_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1000])

    predict = vgg16(x)
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))

    with tf.variable_scope("train"):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        correct_train = tf.equal(tf.arg_max(predict, 1), tf.arg_max(y, 1))
        accuracy_train = tf.reduce_mean(tf.cast(correct_train, "float"))

        with tf.Session() as sess:
            saver = tf.summary.FileWriter("../log/", sess.graph)
            sess.run(tf.global_variables_initializer())
            X_all, y_all = load_data()
            y_all = sess.run(y_all)
            for epoch in tqdm(range(10)):
                epoch_loss = 0
                for i in range(int((len(img_df) - 1) / batch_size) + 1):
                    min_end = min(batch_size * (i + 1), len(img_df))
                    epoch_x, epoch_y = X_all[i * batch_size:min_end, :, :, :], y_all[i * batch_size:min_end, :]
                    _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print("loss", epoch_loss)
