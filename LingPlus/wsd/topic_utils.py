import numpy as np
from scipy.misc import logsumexp

def get_word_topic_prob(word, theta, model, dictionary):
    # initialize theta_vec
    if theta is None or len(theta) == 0:
        # uniform prior
        theta_vec = np.full(model.num_topics, 1/model.num_topics)
    else:
        # prior from argument
        theta_vec = theta
    
    widx = dictionary.doc2bow([word])[0][0]    
    prob_wz = model.get_term_topics(widx, minimum_probability=1e-8)    
    
    wz_vec = np.zeros(model.num_topics, dtype=np.double)
    for wz_x in prob_wz:
        wz_vec[wz_x[0]] = wz_x[1]    
    z_vec = theta_vec * wz_vec
    z_vec /= np.sum(z_vec)    
    return(z_vec)

def get_doc_topic_prob(word_list, model, dictionary):
    model.random_state = np.random.RandomState(15025)
    doc_topics = model.get_document_topics(dictionary.doc2bow(word_list), 
            minimum_probability=1e-5)
    doc_vec = np.zeros(model.num_topics, dtype=np.double)
    for tx in doc_topics:
        doc_vec[tx[0]] = tx[1]
    doc_vec /= np.sum(doc_vec)
    return doc_vec
import pdb


def get_word_assoc(words_y, words_x, theta, model, dictionary):
    """compute p(w1|w2) association probability
    """
    if not isinstance(words_x, list):
        words_x = [words_x]
    if not isinstance(words_y, list):
        words_y = [words_y]

    xidx_vec = [x[0] for x in dictionary.doc2bow(words_x)]    
    yidx_vec = [x[0] for x in dictionary.doc2bow(words_y)]    

    if theta is None or len(theta) == 0:
        theta = np.full(model.num_topics, 1/model.num_topics, dtype=np.double)

    # phi_x is a n_topic x n_words_x matrix
    phi_x = model.expElogbeta[:, xidx_vec]
    logphi_x_zvec = np.sum(np.log(phi_x), 1) + np.log(theta)
    K_words_x = logsumexp(logphi_x_zvec)

    logphi_y = np.log(model.expElogbeta[:, yidx_vec])
    logp_vec = np.zeros(len(yidx_vec), dtype=np.double)
    for yi in range(len(yidx_vec)):
        logp_vec[yi] = logsumexp(logphi_y[:, yi] + logphi_x_zvec)\
                        - K_words_x    
    return np.exp(logp_vec)
