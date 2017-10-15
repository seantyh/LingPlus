import tensorflow as tf
import numpy as np

def model_predict(model, Xvec):
    X0_sum = tf.constant(model[0], name="X1sum")
    X1_sum = tf.constant(model[1], name="X0sum")        
    pY1 = tf.constant(0.5, dtype=tf.double)
    alpha = 1
    X1_param = (X1_sum+alpha)/(tf.reduce_sum(X1_sum)+X1_sum.shape[0]*alpha)
    X0_param = (X0_sum+alpha)/(tf.reduce_sum(X0_sum)+X0_sum.shape[0]*alpha)
    Xdbl = tf.to_double(Xvec)    
    p1_nZ_x = tf.pow(X1_param, Xdbl)
    p0_nZ_x = tf.pow(X0_param, Xdbl)    
    p1_nZ_xsub = tf.boolean_mask(p1_nZ_x, Xvec > 0)
    p0_nZ_xsub = tf.boolean_mask(p0_nZ_x, Xvec > 0)
    log_p1_nZ = tf.log(pY1) + tf.reduce_sum(tf.log(p1_nZ_xsub))
    log_p0_nZ = tf.log(pY1) + tf.reduce_sum(tf.log(p0_nZ_xsub))        
    return [log_p0_nZ, log_p1_nZ]

def predict(model, test_x):      
    feed_x = test_x.toarray()            
    
    pred_vec = []    
    predX = tf.placeholder(name="predX", dtype=tf.int32, shape=(feed_x.shape[1],))
    yhat = model_predict(model, predX)
    sess = tf.Session()

    for row_i in range(feed_x.shape[0]):        
        p_nZ = sess.run(yhat, feed_dict = {predX: feed_x[row_i,:]})
        pred_vec.append(np.argmax(p_nZ))          
    sess.close()

    return(pred_vec)    