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
from app.irsystem.models.shared_variables import min_document_frequency
from app.irsystem.models.inverted_index import InvertedIndex
from app.irsystem.models.processing import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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


def make_inverted_index(data):
    i = InvertedIndex()
    i.create(data)
    return i


"""
Compute IDF values from the inverted index
Returns: idf dictionary {term: idf value}
"""


def get_idf(inv_index, num_docs, min_df=0, max_df=0.1):  # TODO: change these min/max in the future
    idf = {}
    max_rat = max_df * num_docs
    for k, v in inv_index.items():
        df = len(v)
        if df >= min_df and df <= max_rat:
            idf[k] = num_docs / float(df)
            #idf[k] = math.log(num_docs / (1 + float(df)), 1.5)
        else:
            print("idf of {} is {} with a total of {}".format(k, df, num_docs))
    return idf


"""
  Computes norm of each document
  Returns: dict where {doc_id: norm of doc}
"""


def get_doc_norms(inv_index, idf, num_docs):
    norms = {}

    for k, v in inv_index.items():
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

def make_post_subreddit_lookup(data, inverted_index):
    post_lookup = {}
    subreddit_lookup = {}
    sentiment_lookup = {}
    analyzer = SentimentIntensityAnalyzer()
    for post in data:
        #Create sentiment analysis lookup data
        score = analyzer.polarity_scores(post['selftext'])['compound']
        sentiment_lookup[post['id']] = score

        post_lookup[post['id']] = {}
        post_lookup[post['id']]['subreddit'] = post['subreddit']
        cnt = Counter()
        for token in post['tokens']:
            if token in inverted_index:
                cnt[token] += 1
        post_lookup[post['id']]['word_count'] = cnt.most_common()
        if not post['subreddit'] in subreddit_lookup:
            subreddit_lookup[post['subreddit']] = 0
        subreddit_lookup[post['subreddit']] += 1
    return post_lookup, subreddit_lookup, sentiment_lookup

def create_and_store_structures():
    print("...creating structures")

    print("...loading data")
    data = load_data()
    num_docs = len(data)

    #need to tokenize all posts first
    print("tokenizing posts")
    i = 0
    for post in data:
        i += 1
        if i % 50 == 0:
            printProgressBar(i, num_docs)
        post['tokens'] = tokenize(" ".join(post['tokens']), False)


    print("...making inverted index(will take a long time)")
    inverted_index = make_inverted_index(data)

    inverted_index = InvertedIndex()
    inverted_index.load()

    print("...computing idf for {}".format(num_docs))
    idf = get_idf(inverted_index, num_docs, min_document_frequency, max_document_frequency)

    print("...pruning inverted index")
    #remove values that were removed from the idf
    tokens = inverted_index.keys()
    for token in tokens:
        if not token in idf:
            print("removing: {}".format(token))
            inverted_index.remove_token(token)

    print("...making subreddit lookup")
    post_lookup, subreddit_lookup, sentiment_lookup = make_post_subreddit_lookup(data, inverted_index)

    print("...getting doc norms")
    norms = get_doc_norms(inverted_index, idf, num_docs)

    # store data in pickle files
    print("...storing post lookup")
    pickle.dump(post_lookup, open(file_path_name + "-post_lookup.pickle", 'wb'))
    print("...storing subreddit lookup")
    pickle.dump(subreddit_lookup, open(file_path_name + "-subreddit_lookup.pickle", 'wb'))
    print("...storing sentiment lookup")
    pickle.dump(sentiment_lookup, open(file_path_name + "-sentiment_lookup.pickle", 'wb'))
    print("...storing inverted index")
    inverted_index.store()
    print("...storing idf")
    pickle.dump(idf, open(file_path_name + "-idf.pickle", 'wb'))
    print("...storing doc norms")
    pickle.dump(norms, open(file_path_name + "-norms.pickle", 'wb'))
    print("completed creating and storing structures.")

# Print iterations progress (got from online)
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
