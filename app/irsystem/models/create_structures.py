from collections import defaultdict
from collections import Counter
import json
import math
import string
import pickle
import time
import numpy as np
from app.irsystem.models.shared_variables import file_path
from app.irsystem.models.shared_variables import file_path_name
from app.irsystem.models.shared_variables import max_document_frequency
"""
this file is to create datastructures. Currently creates:

- inverted index
"""


def load_data():
    # open the already filtered dataset
    with open(file_path) as file:
        return json.load(file)


"""
  Makes an inverted index from given file (list of posts)
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
            inv_index[k].append((file[i]['id'], temp_dict[k]))
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
  Returns: dict where {doc_id: norm of doc}
"""


def get_doc_norms(inv_index, idf, num_docs):
    norms = {}

    for k, v in inv_index.items():
        if k in idf:
            idf_i = idf[k]
            for pair in v:
                i = pair[0]
                tf = pair[1]
                if i not in norms:
                    norms[i] = (tf*idf_i)**2
                else:
                    norms[i] += (tf*idf_i)**2

    for k, v in norms.items():
        norms[k] = v**(0.5)

    return norms

def make_post_lookup(data):
    post_lookup = {}
    for post in data:
        post_lookup[post['id']] = post['subreddit']
    return post_lookup

def create_and_store_structures():
    print("...creating structures")
    data = load_data()
    num_docs = len(data)

    post_lookup = make_post_lookup(data)
    inverted_index = make_inverted_index(data)
    idf = get_idf(inverted_index, num_docs, 0, max_document_frequency)
    norms = get_doc_norms(inverted_index, idf, num_docs)
    # store data in pickle files
    pickle.dump(post_lookup, open(file_path_name + "-post_lookup.pickle", 'wb'))
    pickle.dump(inverted_index, open(file_path_name + "-inverted_index.pickle", 'wb'))
    pickle.dump(idf, open(file_path_name + "-idf.pickle", 'wb'))
    pickle.dump(norms, open(file_path_name + "-norms.pickle", 'wb'))
    print("completed creating and storing structures.")