import pandas as pd
import re
import pyOceanus
import pdb
from itertools import chain

try:
    oc = pyOceanus.Oceanus()
except Exception as ex:
    print(ex)

cache_dict = {}

def set_Oceanus_Endpoint(url):
    global oc
    oc = pyOceanus.Oceanus(url)

def make_example_data(cwn_data):
    rows = []
    for lemma, senses in cwn_data.items():
        for senseid, senseObj in senses.items():
            for ex_i, ex in enumerate(senseObj["example_cont"]):
                widx = ex.find("<")
                sent = re.sub("[<>'\"]", "", ex)
                sent = sent.strip()
                rows.append((lemma, senseid, widx, ex_i, sent))
                
    sense_data = pd.DataFrame.from_records(rows, 
            columns=["lemma", "senseid", "widx", "exid", "example"])
    return(sense_data)

def segment(text):
    if text in cache_dict:
        od = cache_dict[text]
    else:
        od = oc.parse(text)
        cache_dict[text] = od
    token_data = chain.from_iterable(od.tokens)
    return list(token_data)

def find_word_index(token_data, chidx):
    doc_words = [x[0] for x in token_data]
    doc_chstart = [x[3] for x in token_data]
    widx_diff = [abs(x - chidx) for x in doc_chstart]
    widx_pos = widx_diff.index(min(widx_diff))
    return widx_pos

def find_word_in_text(token_data, target, offset = 0):
    has_occur = [i for i, x in enumerate(token_data) if target in x[0]]
    has_occur = [x for x in has_occur if x >= offset]
    if has_occur:
        return has_occur[0]
    else:
        return None

def preprocess(token_data, widx):            
    doc_words = [x[0] for x in token_data]
    return (widx, doc_words)

def example_word_list(text, chidx):
    token_data = segment(text)
    widx = find_word_index(token_data, chidx)
    return preprocess(token_data, widx)
