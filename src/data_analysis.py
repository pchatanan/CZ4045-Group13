import json
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import multiprocessing
import sys
import string
import pickle

ps = PorterStemmer()  # This is for nltk stemming
stop_words_custom = ['n\'t', 'one', 'two', '\'s', 'would', 'get', 'very']

do_not_stem_list = ["the", "this", "was", "battery", "charge", "because", "very", "verify"]
my_stem_ref = {
    "charged": "charge",
    "charges": "charge",
    "charger": "charge",
    "charging": "charge",
    "batteries": "battery",
    "batterys": "battery",
    "verified": "verify",
    "verifies": "verify"
}


def check_corpus(corpus_name):
    """
    This function checks whether a corpus is available for referencing during Sentence Segmentation,
    Tokenization and POS Tagging. If not available, it will automatically download.
    :param corpus_name: name of the corpus
    """
    try:
        nltk.data.find('corpus_name')
        print("{} is installed".format(corpus_name))
    except LookupError:
        print("{} is not installed. Trying to download now...".format(corpus_name))
        nltk.download(corpus_name)


def my_stem(word):
    """
    This function perform stemming
    :param word: this non-stemmed word
    :return: stemmed word
    """
    if word in do_not_stem_list:
        return word
    if word in my_stem_ref:
        return my_stem_ref[word]
    else:
        return ps.stem(word)


def find_most_frequent(data_list, key, top_n):
    """
    This function receives a list of json objects with keys and corresponding values.
    It counts the number of json objects with identical value of the specified key and sort the result in a list.
    Ex:
    my_list = [{"name": "John"}, {"name": "Marry"}, {"name": "John"}]
    result = find_most_frequent(my_list, "name", None)
    >> result = [("John", 2), ("Marry", 1)]
    :param data_list: a list of json objects
    :param key: the key corresponding to value we want to count
    :param top_n: only return top_n result (if None it will return all the result)
    :return: a list of tuple containing (value, count)
    """
    attr_list = [item[key] for item in data_list]
    result = Counter(attr_list).most_common()
    if top_n is None:
        return result
    if len(result) > top_n:
        return [result[i] for i in range(top_n)]
    return result


def plot_frequency(result,
                   title,
                   x_label,
                   y_label,
                   bar_direction):
    """
    Plot bar graphs
    :param result: a list of tuple from "find_most_frequent" function
    :param title: title of the graph
    :param x_label: x-axis legend
    :param y_label: y-axis legend
    :param bar_direction: "v" for vertical or "h" for horizontal
    """
    # setup the plot
    fig, ax = plt.subplots()
    plt.title(title)
    y = [item[1] for item in result]

    x = [item[0] for item in result]
    if bar_direction == "v":
        y.reverse()
        x.reverse()
        ax.barh(x, y, color="blue")
        for i, v in enumerate(y):
            ax.text(v, i, " " + str(v), color='blue', va='center', fontweight='bold')
        plt.subplots_adjust(left=0.3)
    else:
        ax.bar(x, y, color="blue")
        for i, v in enumerate(y):
            ax.text(i, v, str(v) + "\n", color='blue', va='center', fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('./output/{}.png'.format(title))


def remove_stop_words(counted_list):
    """
    Remove stop words
    :param counted_list: A list of tuple containing (word, count)
    :return: the counted_list with stop words removed
    """
    cleaned_counted_list = [item for item in counted_list if not (item[0].lower() in stop_words_list)]
    return cleaned_counted_list


def process_review(text):
    review = json.loads(text)
    # remove unwanted attr
    del review['unixReviewTime']
    del review['reviewTime']
    sentence_tokenize_list = sent_tokenize(review["reviewText"])
    del review['reviewText']
    # Store result back into review
    review["sentence_tokenize"] = sentence_tokenize_list
    review["num_sentences"] = str(len(sentence_tokenize_list))
    # Word Tokenize and Stemming
    stemmed_words = []
    non_stemmed_words = []
    for sentence in sentence_tokenize_list:
        word_list = word_tokenize(sentence)
        non_stemmed_words += word_list
        for word in word_list:
            stemmed_words.append(my_stem(word.lower()))
    review["non_stemmed_words"] = non_stemmed_words
    review["num_non_stemmed_words"] = str(len(set(non_stemmed_words)))  # Set is needed (unrepeated count)
    review["stemmed_words"] = stemmed_words
    review["num_stemmed_words"] = str(len(set(stemmed_words)))  # Set is needed (unrepeated count)
    return review


def log_result(retrieval):
    results.append(retrieval)
    all_non_stemmed_words.extend(retrieval["non_stemmed_words"])
    all_stemmed_words.extend(retrieval["stemmed_words"])
    sys.stderr.write('\rdone {0:%}'.format(len(results) / n_reviews))


if __name__ == "__main__":
    # Load necessary corpus
    for corpus in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
        check_corpus(corpus)
    stop_words_list = stopwords.words("english") + list(string.punctuation) + stop_words_custom

    # Load data
    print("Loading pickle...")
    try:
        results, all_non_stemmed_words, all_stemmed_words = pickle.load(open("./raw/save.p", "rb"))
        print("Found pickle!")
    except FileNotFoundError:
        print("Data is not processed yet. Processing now...")
        data = []
        # file_name = './raw/SampleReview.json'
        file_name = './raw/CellPhoneReview.json'
        with open(file_name) as file:
            for line in file:
                data.append(line)
        # Sentence Segmentation
        n_reviews = len(data)
        # Declare golbal variable to store result
        results = []
        # Variable for word tokenization
        all_non_stemmed_words = []
        all_stemmed_words = []

        # perform parallel processing (must faster than serial)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        for item in data:
            pool.apply_async(process_review, args=[item], callback=log_result)
        pool.close()
        pool.join()
        print("")  # print linebreak

        # save in pickle
        pickle.dump((results, all_non_stemmed_words, all_stemmed_words), open("./raw/save.p", "wb"))

    # Data set Analysis
    top_10_products = find_most_frequent(results, "asin", 10)
    top_10_reviewers = find_most_frequent(results, "reviewerID", 10)
    top_10_num_sentences = find_most_frequent(results, "num_sentences", 10)
    top_10_num_non_stemmed_words = find_most_frequent(results, "num_non_stemmed_words", 10)
    top_10_num_stemmed_words = find_most_frequent(results, "num_stemmed_words", 10)

    # Print 5 sample sentences (result from sentence segmentation)
    random.seed(0)  # seed to make random the same for consistency
    n_samples = 5
    print("\n{} sampled sentences".format(n_samples))
    i = 0
    while i < 5:
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        if sample_sentence in stop_words_list:  # sometimes sentence segmentation break "!" into a sentence
            i -= 1
        else:
            i += 1
            print(sample_sentence)

    # Count words
    non_stemmed_words_frequency = Counter(all_non_stemmed_words).most_common()
    stemmed_words_frequency = Counter(all_stemmed_words).most_common()

    # Remove Stop Words (defined on top)
    non_stemmed_words_frequency = remove_stop_words(non_stemmed_words_frequency)
    stemmed_words_frequency = remove_stop_words(stemmed_words_frequency)

    print("\nList the top-20 most frequent words")
    print("\nnon_stemmed_words_frequency")
    for i in range(10):
        print(non_stemmed_words_frequency[i])
    print("\nstemmed_words_frequency")
    for i in range(10):
        print(stemmed_words_frequency[i])

    # POS Tagging
    n_samples = 5
    print("\n{} POS Tagging".format(n_samples))
    for i in range(5):
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        word_tokens = word_tokenize(sample_sentence)
        print(nltk.pos_tag(word_tokens))

    # Plotting results
    plot_frequency(top_10_products,
                   "Top 10 products reviewed",
                   "No. of reviews",
                   "asin (product ID)",
                   "v")
    plot_frequency(top_10_reviewers,
                   "Top 10 reviewers",
                   "No. of reviews",
                   "reviewer ID",
                   "v")
    plot_frequency(top_10_num_sentences,
                   "Top 10 Number Sentences",
                   "No. of sentences",
                   "No. of reviews",
                   "h")
    plot_frequency(top_10_num_non_stemmed_words,
                   "Top 10 Number of Non-Stemmed Words (unrepeated)",
                   "No. of words",
                   "No. of reviews",
                   "h")

    plot_frequency(top_10_num_stemmed_words,
                   "Top 10 Number of Stemmed Words (unrepeated)",
                   "No. of words",
                   "No. of reviews",
                   "h")
    plt.show()
