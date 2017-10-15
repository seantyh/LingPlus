import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from .train import train
from .predict import predict

def smooth_selection(sm_alpha, train_x, train_y):
    kfold = StratifiedKFold(n_splits=5)
    kfold_i = 0
    acc_vec = []
    lbin = LabelBinarizer()
    lbin.fit(train_y)

    for train_idx, test_idx, in kfold.split(train_x, train_y):
        # print("CV-train: ", train_idx[:10])
        # print("CV-test: ", test_idx[:10])    
        cv_train_x = train_x[train_idx]
        cv_train_y = train_y[train_idx]        
        cv_train_y = lbin.transform(cv_train_y)
        pY = cv_train_y.sum() / len(cv_train_y)
        # print("prior p(Y):", pY)
        cv_test_x = train_x[test_idx]
        cv_test_y = train_y[test_idx]

        # model training
        with tf.Graph().as_default() as g:
            model = train(cv_train_x, cv_train_y,
                            pY_prior=pY, smooth_alpha=sm_alpha)

        # model evaluation
        with tf.Graph().as_default() as g:
            cv_pred_vec = predict(model, cv_test_x)

        cv_pred_y = lbin.inverse_transform(np.array(cv_pred_vec))
        acc = accuracy_score(cv_test_y, cv_pred_y)
        # print("%d: Accuracy: %.4f" % (kfold_i, acc))
        acc_vec.append(acc)
        kfold_i += 1    
    print("Accuracy average: %.4f" % np.mean(acc_vec))
    return(np.mean(acc_vec))