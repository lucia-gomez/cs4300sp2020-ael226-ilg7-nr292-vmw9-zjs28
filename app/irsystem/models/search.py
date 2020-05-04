import pickle
import json
import time
from collections import Counter
from app.irsystem.models.create_dataset import create_dataset, create_description_dataset
from app.irsystem.models.processing import process_data
from app.irsystem.models.create_structures import create_and_store_structures
from app.irsystem.models.shared_variables import file_path
from app.irsystem.models.shared_variables import file_path_name
from app.irsystem.models.comparison import compare_string_to_posts
from app.irsystem.models.comparison import find_subreddits
from app.irsystem.models.inverted_index import InvertedIndex


class SearchEngine():
    def __init__(self, should_build_structures):
        if should_build_structures:
            self.create()
        self.init_pickles()

    def init_pickles(self):
        idf, norms, post_lookup, subreddit_lookup, descriptions, sentiment_lookup = self.open_datastructures()
        self.inverted_index = None
        self.idf = idf
        self.norms = norms
        self.post_lookup = post_lookup
        self.subreddit_lookup = subreddit_lookup
        self.descriptions = descriptions
        self.sentiment_lookup = sentiment_lookup

    def open_datastructures(self):
        with open(file_path_name + "-idf.pickle", 'rb') as file:
            print("...loading idf")
            idf = pickle.load(file)
            print("num words in idf: {}".format(len(idf.keys())))
            print("finished loading idf.")

        with open(file_path_name + "-norms.pickle", 'rb') as file:
            print("...loading norms")
            norms = pickle.load(file)
            print("finished loading norms")

        with open(file_path_name + "-post_lookup.pickle", 'rb') as file:
            print("...loading posts")
            post_lookup = pickle.load(file)
            print("# of posts: " + str(len(post_lookup.keys())))
            print("finished loading posts")

        with open(file_path_name + "-subreddit_lookup.pickle", 'rb') as file:
            print("...loading posts")
            subreddit_lookup = pickle.load(file)
            print("finished loading posts")

        with open(file_path_name + "-descriptions.pickle", 'rb') as file:
            print("...loading descriptions")
            descriptions = pickle.load(file)
            print("finished loading descriptions")

        with open(file_path_name + "-sentiment_lookup.pickle", 'rb') as file:
            print("...loading sentiment scores")
            sentiment_lookup = pickle.load(file)
            print("finished loading sentiment scores")

        with open(file_path_name + "-avg_sentiment.pickle", 'rb') as file:
            print("...loading avg sentiment scores")
            avg_sentiment_lookup = pickle.load(file)
            print("finished loading avg sentiment scores")

        return idf, norms, post_lookup, subreddit_lookup, descriptions, avg_sentiment_lookup

    def run_tests(self):
        print("Testing for sentiment weighting: ")
        self.inverted_index = InvertedIndex()
        self.inverted_index.load()
        queries = [
            "I've been playing Animal Crossing on my Nintendo Switch during quarantine - does anyone have any game recommendations?",
            "Hey Reddit, my girlfriend killed my cat and I broke up with her over text. All my friends are mad at me and said I'm a jerk. Am I the asshole?",
            "I (30M) have been fighting with my wife (29F) for a few days. I have been getting home late from work due to overtime and she's been getting on my case about chores I have to do that she doesn't. AITA?",
            "Niantic released an announcement saying that this week's community day will need an exclusive raid pass. Does anyone know where to get one? I really want to catch some shines or legendaries.",
            "I've been trying to decide what Harry Potter house I belong in.  I identify most with Hermione Granger so I might be a Gryffindor, but to be honest I'm not sure I'm brave enough. I'm a bit nerdy, so maybe Ravenclaw would fit better?  How can I know for sure?"]
        relevant = [
            ["AnimalCrossing", "NintendoSwitch", "Nintendo"],
            ["AmItheAsshole", "relationship_advice", "AskMen"],
            ["AmItheAsshole", "relationship_advice", "relationships"],
            ["pokemongo", "TheSilphRoad", "pokemon"],
            ["harrypotter", "FanTheories", "movies"]]
        scores = {}
        for i in [x * 0.05 for x in range(0, 21)]:
            for j in range(len(queries)):
                ranks = compare_string_to_posts(
                    self.inverted_index, self.idf, self.norms, self.post_lookup, self.sentiment_lookup, i, queries[j], "")
                subreddits = find_subreddits(
                    20, ranks, self.post_lookup, self.subreddit_lookup, self.descriptions)
                for k in range(20):
                    if subreddits[k]["subreddit"] in relevant[j]:
                        if i in scores:
                            scores[i] += 1 / (k+1)
                        else:
                            scores[i] = 1 / (k+1)
        print("TESTING COMPLETED\nRESULTS:")
        print(scores)

    def search(self, query, title):
        if self.inverted_index is None:
            self.inverted_index = InvertedIndex()
            self.inverted_index.load()
        ranks = compare_string_to_posts(self.inverted_index,
                                        self.idf, self.norms, self.post_lookup, self.sentiment_lookup, 0.25, query, title)
        return find_subreddits(10, ranks, self.post_lookup, self.subreddit_lookup, self.descriptions)

    def create(self):
        # create the dataset from the pushshift api
        print("Looking at " + file_path_name)
        print(
            "create dataset? this will make queries to the api and can take a long time y/n")
        ans = input()
        if ans == 'y':
            create_dataset()

        print("create description dataset? this will make queries to the api and can take a long time y/n")
        ans = input()
        if ans == 'y':
            create_description_dataset()

        # create the idf, inverted index, and norms

        print("create and store structures? y/n")
        ans = input()
        if ans == 'y':
            create_and_store_structures()

        print("Run interactive tests? y/n")
        if input() == 'y':
            self.init_pickles()
            self.run_tests()

        print("delay end.")
        input()
