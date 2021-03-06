import os

this_dir = os.path.dirname(os.path.abspath(__file__))
jar = this_dir + "/picklejar/"
num_posts = 500
num_subreddits = 1075
min_words_per_post = 15
num_partitions = 50 #for the inverted index
max_document_frequency = 0.10
min_document_frequency = 15
create_dataset_or_structures = False
pseudo_relevance_rocchio_top_posts = 15
pseudo_relevance_rocchio_lowest_posts = 50
"""
DATASET NAMING CONVENTION
""<num posts>p<num subreddits>s<min words per post>mwc.json"
Example: 1000p700s10mwc.json
"""
file_path_name = jar + str(num_posts) + 'p' + str(num_subreddits) + 's' + str(min_words_per_post) + 'mwc'
file_path = file_path_name + ".json"
reddit_list = jar + 'subreddits.csv'
