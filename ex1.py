"""
    NLP - ex1
    Ram Elias 205445794
    Eliezer Seror 312564776
"""

import copy
import re
import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
import scipy


# tokenize the data
def data_fixed(file):
    text = file.readlines()

    for i in range(len(text)):
        sentence = "<S>"
        arr = text[i].split("\t")
        word_arr = arr[1].split(" ")

        for w in word_arr:
            sentence += check_word(w) + ' '

        text[i] = sentence.strip() + " </S>\n"

    return text


# check any word and replace with the correct token
def check_word(word):
        word = re.sub(r'[^\w\s]', '', word)
        word = word.lower()
        word = re.sub(r'([0-9]+)', '<!DIGIT!>', word)
        if re.search(r'[^\x00-\x7F]+', word):
            return '<UNK>'

        return word


# create an empty matrix
def create_matrix():
    rows = get_simlex_words()
    cols = get_freq_words()

    matrix = [cols]

    for i in rows:
        matrix.append([i]+[0] * (len(cols)-1))

    return matrix


# get the words from the simlex without duplicates
def get_simlex_words():
    rows_arr = []
    simlex = open("EN-SIMLEX-999.txt", 'r', encoding="utf8")
    rows = simlex.readlines()

    for i in rows:
        arr = i.split('\t')
        for j in range(len(arr)-1):
            if arr[j] in rows_arr:
                continue
            rows_arr.append(arr[j])

    return rows_arr

# get the 20K frequent word in the english language
def get_freq_words():
    cols_arr = []
    freq = open("eng_wikipedia_2016_1M-words.txt", 'r', encoding="utf8")
    cols = freq.readlines()
    cols_arr.append("-")
    for i in range(50, 20049):  # 50, 20049
        arr = cols[i].split('\t')
        cols_arr.append(arr[1])

    return cols_arr


# Go over text line by line -> pick a word and add all the "size" words to her dict
def make_hash_table(text, size):
    hash_t = {}
    amount_of_words = 0
    for i, sentence in enumerate(text):
        sentence = sentence.split()[1:]
        for j, word in enumerate(sentence):
            amount_of_words += 1
            if word not in hash_t:
                hash_t[word] = {}
                hash_t[word]["$sum"] = 0
            hash_t[word]["$sum"] += 1
            for index in range(-size, size+1):
                if 0 <= j + index < len(sentence) and (index != 0):
                    if sentence[j + index] not in hash_t[word]:
                        hash_t[word][sentence[j + index]] = 0
                    hash_t[word][sentence[j + index]] += 1
    hash_t["amount_of_words"] = amount_of_words
    return hash_t


# fill the table from the 1200 words and 20,000 words with matrix_hash from text
def frequency_counts(matrix, matrix_hash):
    for row in range(1, len(matrix)):
        for col in range(1, len(matrix[0])):
            matrix[row][col] = count(matrix[0][col], matrix[row][0], matrix_hash)
    return matrix


# get the number of shows for 2 words
def count(simelx, wiki, matrix_hash):
    counter = 0
    if simelx in matrix_hash:
        if wiki in matrix_hash[simelx]:
            counter = matrix_hash[simelx][wiki]
    return counter

# PPMI calculation
def calc_ppmi(frequency_matrix, hash_matrix):

    ppmi_matrix = frequency_matrix
    amount_of_words = hash_matrix["amount_of_words"]

    for row in range(1, len(ppmi_matrix)):
        for col in range(1, len(ppmi_matrix[row])):
            if ppmi_matrix[row][col] != 0:
                numerator = (2+ppmi_matrix[row][col])/(amount_of_words + (2 * len(ppmi_matrix[0]) * len(ppmi_matrix)))

                denominator = (hash_matrix[ppmi_matrix[row][0]]["$sum"] / amount_of_words) * \
                              (hash_matrix[ppmi_matrix[0][col]]["$sum"] / amount_of_words)

                ppmi_matrix[row][col] = max(math.log2(numerator/denominator), 0)

    return ppmi_matrix


# cosine measure for similarity between two words from simlex
def cosine_measure(file_name, matrix):
    data = pd.DataFrame(matrix[1:len(matrix)])
    data.index = data[0]
    file = open(file_name, 'w+')
    simlex = open('EN-SIMLEX-999.txt', 'r', encoding="utf8")
    lines = simlex.readlines()

    for line in lines:
        line = line.split('\t')
        word1_vec = data.loc[line[0]].tolist()[1:]
        word2_vec = data.loc[line[1]].tolist()[1:]

        result = np.dot(word1_vec, word2_vec)
        if norm(word1_vec)*norm(word2_vec) == 0:
            result = 0
        else:
            result = result/(norm(word1_vec)*norm(word2_vec))

        file.write(line[0] + '\t' + line[1] + '\t' + str(result) + "\n")


# get the result from the file
def get_file_results(file):
    list1 = []
    simlex = open(file, 'r', encoding="utf8")
    lines = simlex.readlines()

    for line in lines:
        line = line.strip()
        line = line.split('\t')
        list1.append(float(line[2][:4]))

    return list1

# calc the correlation using spearmenr function
def calc_correlation(file, simlex_list):
    model_list = get_file_results(file)
    print(scipy.stats.spearmanr(simlex_list, model_list))


if __name__ == "__main__":
    file = open("eng_wikipedia_2016_1M-sentences.txt", 'r+', encoding="utf8")
    text = data_fixed(file)
    matrix = create_matrix()
    matrix_hash_2 = make_hash_table(text, 2)
    matrix_hash_5 = make_hash_table(text, 5)

    fc2 = frequency_counts(copy.deepcopy(matrix), matrix_hash_2)
    fc5 = frequency_counts(copy.deepcopy(matrix), matrix_hash_5)
    ppmi2 = calc_ppmi(copy.deepcopy(fc2), matrix_hash_2)
    ppmi5 = calc_ppmi(copy.deepcopy(fc5), matrix_hash_5)

    cosine_measure('freq_window2.txt', fc2)
    cosine_measure('freq_window5.txt', fc5)
    cosine_measure('ppmi_window2.txt', ppmi2)
    cosine_measure('ppmi_window5.txt', ppmi5)

    simlex_results = get_file_results('EN-SIMLEX-999.txt')
    print("frequency window 2 result:")
    calc_correlation('freq_window2.txt', simlex_results)
    print("frequency window 5 result:")
    calc_correlation('freq_window5.txt', simlex_results)
    print("ppmi window 2 result:")
    calc_correlation('ppmi_window2.txt', simlex_results)
    print("ppmi window 5 result:")
    calc_correlation('ppmi_window5.txt', simlex_results)



