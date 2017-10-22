import math
import numpy as np
import pdb


class NameDetection:
    def __init__(self, unigrams, bigrams):
        self.ug = unigrams
        self.bg = bigrams
        self.alphaParam = 0.5
        self.ugZ = sum(unigrams.values()) + self.alphaParam*len(unigrams)
        self.bgZ = sum(bigrams.values()) + self.alphaParam*len(bigrams)
        self.lamParam = 0.7
        

    def setLambda(self, lam):
        self.lamParam = lam

    def setAlpha(self, alpha):
        self.alphaParam = alpha
        self.ugZ = sum(unigrams.values()) + self.alphaParam*len(unigrams)
        self.bgZ = sum(bigrams.values()) + self.alphaParam*len(bigrams)
    
    def update_criterion(self, trainx, perct=90):
        probs = [np.min(self.compute_transName_prob(x)) for x in trainx]
        crit = np.percentile(probs, perct)
        self.crit = crit
        return crit

    def compute_transName_prob(self, text):
        f_c0 = self.ug.get(text[0], 0) + self.alphaParam
        prob_vec = [math.log(f_c0/self.ugZ)]
        for c1, c2 in zip(text, text[1:]):
            f_c2 = self.ug.get(c2, 0) + self.alphaParam
            f_c12 = self.bg.get(c1+c2, 0) + self.alphaParam
            prob = self.lamParam * (f_c12/self.bgZ) +\
                   (1-self.lamParam) * (f_c2/self.ugZ)
            lprob = math.log(prob)
            prob_vec.append(lprob)
        return prob_vec

    def get_names(self, text):
        # set length 1 as trivially empty name set
        if len(text) == 1:
            return []

        vec = np.array(self.compute_transName_prob(text))
        name_mask = np.int8(vec > self.crit)        
        # get rid of only-one instance
        for idx in range(len(name_mask)):
            if idx == 0:
                if name_mask[0] == 1 and name_mask[1] == 0:
                    name_mask[0] = 0
            elif idx == len(name_mask) - 1:
                if name_mask[-1] == 1 and name_mask[-2] == 0:
                    name_mask[-1] = 0
            else:
                if name_mask[idx-1] == 0 and name_mask[idx+1] == 0 and \
                   name_mask[idx] != 0:
                   name_mask[idx] = 0
        
        # group the 1's  
        if np.any(name_mask == 1):
            cur = np.min(np.where(name_mask == 1))-1                    
        else:
            cur = len(name_mask)
        name_idx_vec = []
        buf = []
        for idx in range(cur+1, len(name_mask)):
            if name_mask[idx] == 1:
                if idx - cur == 1:                                    
                    buf.append(idx)                    
                else:
                    if buf:
                        name_idx_vec.append(buf)
                        buf = [idx]
                cur = idx
        if buf:
            name_idx_vec.append(buf)        

        name_vec = []
        for idx_vec in name_idx_vec:
            name_vec.append("".join(text[i] for i in idx_vec))
        
        return name_vec





