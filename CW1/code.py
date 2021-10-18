import heapq
import os
import urllib.request
import xml.etree.ElementTree as ElementTree

import numpy as np
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()
stop_words = []
words_positions = {}
inverted_index = {}
df_words = {}
boolean_results = []
ranked_results = []


def download_file_and_save(url, file_name):
    """Download the file from `url` and save it locally under `file_name`:
    Args:
        url (string): The url to download the file from
        file_name (string): The name of the file
    """
    if not os.path.exists('./' + file_name):
        print('Downloading file from {}...'.format(url))
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()
            print('Saving file as {}...'.format(file_name))
            out_file.write(data)


def create_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def load_xml(xml_file):
    with open(xml_file, 'r') as xml_file:  # Read xml file
        xml = xml_file.read()

    return ElementTree.fromstring(xml)


def to_lower_case(input_tokens):
    return [token.lower() for token in input_tokens]


def remove_stopwords(input_tokens):
    return [token for token in input_tokens if token not in stop_words]


def apply_stemming(input_tokens):
    return [ps.stem(token) for token in input_tokens]


def tokenize(input_text):
    text_tokens = re.sub(r"[^\w\s]|_", " ", input_text).split()
    return to_lower_case(text_tokens)


def preprocess(input_text, doc_number):
    """
    input_text = document content (string)
    document_number = extracted from the XML file
    :return: list of (word, document_number, position)
    """

    # split text by whitespace into words by replacing first any non-letter / non-digit character with a space
    # \w matches any [a-zA-Z0-9_] characters
    # \s matches any whitespace characters
    words = re.sub(r"[^\w\s]|_", " ", input_text).split()

    # transform the text to lowercase
    words = to_lower_case(words)

    # remove the stop words
    text_without_sw = remove_stopwords(words)

    # apply stemming
    stemmed_words = apply_stemming(text_without_sw)

    # list of tuples [(word, position)]
    return list(zip(stemmed_words, range(0, len(stemmed_words))))


def create_inverted_index():
    for docId, values in words_positions.items():
        for (word, position) in values:
            if not inverted_index.get(word):
                inverted_index[word] = dict()
            if not inverted_index[word].get(docId):
                inverted_index[word][docId] = []
            inverted_index[word][docId].append(position)


# def process_boolean_queries(queries):
#     logical_operators_mapping = {'and': '&', 'or': '|', 'not': '~'}


def process_ranked_queries(queries, number_docs):
    remove_initial_chars = " 0123456789"
    query_list = [query.lstrip(remove_initial_chars) for query in queries]

    for query in query_list:
        query_terms = apply_stemming(remove_stopwords(to_lower_case(query.split())))
        tfidf_scores = []
        for document in range(1, number_docs):
            tfidf_scores.append(TFIDF(document, query_terms, number_docs))
        top_tfidf = list(heapq.nlargest(150, range(len(tfidf_scores)), key=lambda x: tfidf_scores[x]))
        top_results = [(query, index, top_tfidf[index]) for index in range(1, len(top_tfidf))]
        ranked_results.extend(top_results)


def TFIDF(docId, terms, number_docs):
    """Calculates the retrieval score using the TFIDF (term frequency - inverse document frequency) formula
    Args:
        docId (str)
        terms (list)
        number_docs (list): Total number of documents
    Returns:
        total_score (float): Retrieval score for a query and a document
    """
    total_score = 0

    # For each term calculate the tf (term frequency in doc) and df (number of docs that word appeared in)
    for term in terms:
        if docId in inverted_index[term].keys():
            # Frequency of term in this document
            tf = len(inverted_index[term][docId])
            # Number of documents in which the term appeared
            df = len(inverted_index[term])
            term_weight = (1 + np.log10(tf)) * np.log10(number_docs / df)
            total_score += term_weight

    return total_score


def save_inverted_index_txt(file_name):
    with open(file_name + '.txt', 'w+') as f:
        sorted_words = sorted(inverted_index.keys())

        for word in sorted_words:
            indices_dict = inverted_index[word]
            df_words[word] = len(inverted_index[word])
            f.write(word + ': ' + str(len(inverted_index[word])) + '\n')

            for doc_num in indices_dict:
                indices_str = ', '.join(map(str, indices_dict[doc_num]))
                f.write('\t' + str(doc_num) + ': ' + indices_str + '\n')
            f.write('\n')
        print('Inverted index saved at {}.txt\n'.format(file_name))


def save_boolean_search_results(boolean_queries, file_name):
    with open(file_name + '.txt', 'w+') as f:
        for i, query in enumerate(boolean_queries):
            for docId in boolean_results[i]:
                f.write(str(i + 1) + ',' + docId + '\n')

        print('Boolean search results saved at {}.txt\n'.format(file_name))


def save_ranked_retrieval_results(file_name):
    with open(file_name + '.txt', 'w+') as f:
        for result in ranked_results:
            printed_res = str(result()[0]) + ' 0 ' + result()[1] + ' 0 ' + '%.4f' % result()[2] + ' 0 \n'
            f.write(printed_res)

        print('Ranked search results saved at {}.txt'.format(file_name))


if __name__ == '__main__':
    DATA_DIR = 'data/'

    STOP_WORDS_FILE = 'stop_words.txt'
    TREC_SAMPLE_FILE = DATA_DIR + 'trec.sample.xml'
    QUERIES_BOOLEAN = DATA_DIR + 'queries/boolean.txt'
    QUERIES_RANKED = DATA_DIR + 'queries/ranked.txt'

    RESULTS_DIR = 'results/'
    INVERTED_INDEX_FILE = RESULTS_DIR + 'index'
    RESULTS_BOOLEAN_FILE = RESULTS_DIR + 'boolean'
    RESULTS_RANKED_FILE = RESULTS_DIR + 'ranked'

    download_file_and_save(
        'http://members.unine.ch/jacques.savoy/clef/englishST.txt', STOP_WORDS_FILE)

    # store the list of english stop words
    with open(STOP_WORDS_FILE) as f:
        stop_words = [word.strip() for word in f]

    # store the boolean queries in a list
    with open(QUERIES_BOOLEAN) as f:
        boolean_queries = f.readlines()
        boolean_queries = [boolean_query.rstrip() for boolean_query in boolean_queries]

    # store the ranked queries in a list
    with open(QUERIES_RANKED) as f:
        ranked_queries = f.readlines()
        ranked_queries = [ranked_query.rstrip() for ranked_query in ranked_queries]

    # load trec_sample.xml
    root = load_xml(TREC_SAMPLE_FILE)
    number_docs = 0

    for doc in root:
        doc_Id = doc.find('DOCNO').text
        headline = doc.find('HEADLINE').text
        text = doc.find('TEXT').text
        headline_and_text = headline + ' ' + text
        number_docs += 1

        words_positions[doc_Id] = preprocess(headline_and_text, doc_Id)

    create_directory(RESULTS_DIR)
    create_inverted_index()
    process_ranked_queries(ranked_queries, number_docs)

    save_inverted_index_txt(INVERTED_INDEX_FILE)
    # save_boolean_search_results(boolean_queries, RESULTS_BOOLEAN_FILE)
    save_ranked_retrieval_results(RESULTS_RANKED_FILE)
