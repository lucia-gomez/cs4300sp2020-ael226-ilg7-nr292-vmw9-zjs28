from collections import defaultdict
from collections import Counter
import json
import math
import string
import pickle
import time
import numpy as np
# from nltk.tokenize import TreebankWordTokenizer
# from IPython.core.display import HTML
file_path_name = 'reddit-datasets/reddit-data-1-post-processed'
file_path = file_path_name + ".json"

"""
this file is to create datastructures. Currently creates:

- inverted index
"""

# LOAD DATA
with open(file_path) as file:
    data = json.load(file)

num_docs = len(data)

"""
  Makes an inverted index from given file
  Returns: inverted index as a dictionary
  list of tuples of (post_id, term_count)
"""


def make_inverted_index(file):
    inv_index = {}

    for i in range(0, len(file)):
        temp_dict = {}
        words = file[i]['tokens']
        for w in words:
            if w not in temp_dict:
                temp_dict[w] = 1
            else:
                temp_dict[w] += 1
        for k, v in temp_dict.items():
            if k not in inv_index:
                inv_index[k] = []
            inv_index[k].append((i, temp_dict[k]))
    return inv_index



"""
Compute IDF values from the inverted index
Returns: idf dictionary {term: idf value}
"""


def get_idf(inv_index, num_docs, min_df=0, max_df=1):  # TODO: change these min/max in the future
    idf = {}
    max_rat = max_df * num_docs

    for k, v in inv_index.items():
        df = len(v)
        if df >= min_df and df <= max_rat:
            idf[k] = num_docs / float(df)
    return idf



"""
  Computes norm of each document
  Returns: np.array where norms[i] = norm of document i
"""


def get_doc_norms(inv_index, idf, num_docs):
    norms = np.zeros(num_docs)

    for k, v in inv_index.items():
        if k in idf:
            idf_i = idf[k]
            for pair in v:
                i = pair[0]
                tf = pair[1]
                norms[i] += (tf*idf_i)**2

    norms = np.sqrt(norms)

    return norms



inverted_index = make_inverted_index(data)
idf = get_idf(inverted_index, num_docs)
norms = get_doc_norms(inverted_index, idf, num_docs)

jar = "reddit-datasets/picklejar/"
pickle.dump(inverted_index, open(jar + "inverted_index.pickle", 'wb'))
pickle.dump(idf, open(jar + "idf.pickle", 'wb'))
pickle.dump(norms, open(jar + "norms.pickle", 'wb'))