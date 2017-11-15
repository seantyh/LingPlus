import logging
import numpy as np
from scipy.misc import logsumexp
from . import topic_utils as tu
from . import cwn_preproc as cpre
import pdb

class TopicWSD:
    
    def __init__(self, cwn, model, dictionary):
        self.cwn = cwn
        self.sense_data = self.load_sense_data(cwn)
        self.model = model
        self.dictionary = dictionary
        self.sense_alpha = -1

    def load_sense_data(self, cwn):
        sense_data = cpre.make_example_data(cwn)
        sense_data = sense_data.loc[sense_data["lemma"] != "null", :]
        sense_data.sort_values(["lemma", "senseid"])
        sense_data.index = [sense_data["lemma"], sense_data["senseid"], sense_data["exid"]]
        sense_data = sense_data.drop(["lemma", "senseid", "exid"], axis=1)
        sense_data = sense_data[sense_data.example.map(lambda x: len(x)>0)]
        return sense_data
        
    def show_word_topics(word, theta=None):
        z_vec = tu.get_word_topic_prob(word, theta, tm, dictionary)
        self.print_topic(z_vec)
        # plt.bar(range(tm.num_topics), height = z_vec)
    
    def print_topic(topic_probs):
        if isinstance(topic_probs, int):
            topic_data = [(topic_ids, np.NaN)]
        else:
            sort_ids = np.argsort(-np.array(topic_probs))[:5]
            topic_data = [(i, topic_probs[i]) for i in sort_ids if topic_probs[i] > 0.001]
    
        for tid, tprob in topic_data:
            print("[%3d] %s: %.4f" % (tid, topic_heads[tid], tprob))

    def get_word_index(self, x):
        try:
            return self.dictionary.doc2bow([x])[0][0]
        except IndexError:
            return None    
        
    def get_anchor_topics(self, word_list, widx):        
        target_pos = widx
        doc_topics = tu.get_doc_topic_prob(word_list, self.model, self.dictionary)    
        
        anchor_word = []
        n_neigh = 3
        for wpos in range(target_pos-n_neigh, target_pos+n_neigh+1):
            if wpos == target_pos: continue
            pos = min(len(word_list)-1, max(0, wpos))        
            anchor_word.append(word_list[pos])

        return (doc_topics, anchor_word)

    def get_compatability(self, ambig_data, ref_data):        
        ambig_theta, ambig_word = self.get_anchor_topics(ambig_data[1], ambig_data[0])
        ref_theta, ref_word = self.get_anchor_topics(ref_data[1], ref_data[0])
        if not ref_word or not ambig_word:
            return []
        comp_score = tu.get_word_assoc(ambig_word, ref_word, ref_theta, 
                        self.model, self.dictionary)
        
        return(comp_score) 
    
    def has_lemma(self, lemma):
        return lemma in self.sense_data.index.get_level_values(0)

    def get_sense_def(self, lemma, senseid):
        sense_def = self.cwn.get(lemma, {}).get(senseid, {}).get("sense_def", None)
        return sense_def        

    def get_lemma_senses(self, lemma):
        lemma_data = self.sense_data.loc[lemma]
        sense_list = lemma_data.index.get_level_values(0).unique()
        return(sense_list)
    
    def get_sense_example(self, lemma, senseid, exid):
        exdata = self.sense_data.loc[lemma, senseid, exid]
        return cpre.example_word_list(exdata[1], exdata[0])

    def get_disambiguate_refdata(self, lemma):
        HELDOUT_IDX = 1
        try:
            lemma_data = self.sense_data.loc[lemma]
        except IndexError:
            return None
        
        mask = lemma_data.index.map(lambda x: x[1] != HELDOUT_IDX).values
        ref_data = lemma_data[mask]
        senseids = np.unique([x[0] for x in ref_data.index.values])

        ref_list = []
        for senseid in senseids:
            ref_vals = []
            for ref_x in ref_data.loc[senseid, ["widx", "example"]].values:
                ref_vals.append(cpre.example_word_list(ref_x[1], ref_x[0]))
            ref_list.append((senseid, ref_vals))
        return ref_list

    def word_sense_disambiguate(self, word_list, widx, lemma):
        ref_list = self.get_disambiguate_refdata(lemma)
        ambig_data = [widx, word_list]
        n_ref_senses = len(ref_list)
        if self.sense_alpha < 0:
            sense_priors = np.full(n_ref_senses, 1/n_ref_senses, dtype=np.double)
        else:
            w = np.exp(-np.arange(n_ref_senses)/(self.sense_alpha * n_ref_senses))
            sense_priors = w / np.sum(w)            
                
        log_ptilde_sense = np.zeros(n_ref_senses, dtype=np.double)
        for sense_i, (senseid, ref_examples) in enumerate(ref_list):
            log_prob_wassoc = -500
            # assume uniform prior w.r.t examples
            example_prior = 1 / len(ref_examples)
            for ex_i, ref_data in enumerate(ref_examples):
                logging.getLogger().debug("-- %s, %s --", senseid, str(ref_data[1]))

                compat_scores = self.get_compatability(ambig_data, ref_data)
                logscore = np.log(sense_priors[sense_i]) + np.sum(np.log(compat_scores))                
                log_prob_wassoc = np.logaddexp(log_prob_wassoc, logscore)                
                logging.getLogger().debug("[%s-%d], logP: %.4f" % (senseid, ex_i, logscore))                
            
            log_ptilde_sense[sense_i] = log_prob_wassoc
            logging.getLogger().debug("[%s], logPtilde_sense: %.4f" % (senseid, log_ptilde_sense[sense_i]))

        log_p_sense = log_ptilde_sense - logsumexp(log_ptilde_sense)                

        return [(sid, prob) for sid, prob in 
                zip([x[0] for x in ref_list], np.exp(log_p_sense))]

    def get_best_sense(self, psense):
        probs = [x[1] for x in psense]
        return psense[np.argmax(probs)][0]
