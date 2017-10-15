import tensorflow as tf
import numpy as np

def update_model(X, y, nfeat):    
    X1_sum = tf.Variable(tf.zeros((nfeat,), dtype=tf.int32), name='X1Sum')
    X0_sum = tf.Variable(tf.zeros((nfeat,), dtype=tf.int32), name='X0Sum')

    y1_idx = tf.where(tf.equal(y, 1))[:, 0]
    y0_idx = tf.where(tf.equal(y, 0))[:, 0]    
    tf.summary.scalar('y1_idx_sum', tf.reduce_sum(y1_idx))
    X0 = tf.gather(X, y0_idx)
    X1 = tf.gather(X, y1_idx)    
    X0_sum = tf.assign_add(X0_sum, tf.reduce_sum(X0, 0))    
    X1_sum = tf.assign_add(X1_sum, tf.reduce_sum(X1, 0))
    return [X0_sum, X1_sum]

def train(train_x, train_y):
    nfeat = train_x.shape[1]
    X = tf.placeholder(tf.int32, shape=(None, nfeat), name='dataX')
    y = tf.placeholder(tf.int32, shape=(None, 1), name='dataY')            
    
    sess = tf.Session()
    train_op = update_model(X, y, nfeat)
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("h:/train_tb", sess.graph)
    cvM = train_y.shape[0]
    nBatch = 1000

    for row_i in range(0, cvM, nBatch):
        ## print("now in %d, total %d" % (row_i, train_y.shape[0]))
        feed_y = train_y[row_i: row_i + nBatch, :]
        # note sparse matrix doesn't automatic ignore out-of-bound index
        feed_x = train_x[row_i: min(row_i + nBatch, cvM), :].toarray()    
        summary, model = sess.run([merged, train_op],     
                                    feed_dict={X: feed_x, y: feed_y})    
        train_writer.add_summary(summary, row_i)
    # print(sess.run([tf.reduce_sum(model[0]), tf.reduce_sum(model[1])]))
    sess.close()
    return model

