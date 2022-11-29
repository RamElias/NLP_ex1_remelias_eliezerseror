import re
import numpy as np
import math


def data_fixed(file):
    text = file.readlines()

    for i in range(len(text)):
        sentence = "<S>"
        arr = text[i].split("\t")
        word_arr = arr[1].split(" ")

        for w in word_arr:
            sentence += check_word(w) + ' '

        text[i] = sentence.strip() + "</S>\n"

    return text


def check_word(word):
        word = re.sub(r'[^\w\s]', '', word)
        word.lower()
        word = re.sub(r'([0-9]+)', '<!DIGIT!>', word)
        if re.search(r'[^\x00-\x7F]+', word):
            return '<UNK>'

        return word


def create_matrix():
    rows = get_rows()
    cols = get_cols()

    matrix = []
    matrix.append(cols)

    for i in rows:
        matrix.append([i]+[0] * (len(cols)-1))

    return matrix


def get_rows():
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


def get_cols():
    cols_arr = []
    freq = open("eng_wikipedia_2016_1M-words.txt", 'r', encoding="utf8")
    cols = freq.readlines()
    cols_arr.append("#")
    for i in range(50, 100):
        arr = cols[i].split('\t')
        cols_arr.append(arr[1])

    return cols_arr


def make_hash_table(text):
    hash_t = {}

    for i, sentence in enumerate(text):
        sentence = sentence[sentence.index("<S>") + len("<S>"):sentence.index("</S>") - 1].split(" ")

        for j, word in enumerate(sentence):
            if word in hash_t:
                hash_t[word].append([i, j])
            else:
                hash_t[word] = [[i, j]]

    return hash_t


def frequency_counts(matrix, matrix_hash, size):

    for row in range(1, len(matrix)):
        for col in range(1, len(matrix[0])):
            matrix[row][col] = count(matrix[0][col], matrix[row][0], matrix_hash, size)
    return matrix


def count(simelx, wiki, matrix_hash, size):
    counter = 0
    if simelx in matrix_hash and wiki in matrix_hash:
        for i in matrix_hash[simelx]:
            for j in matrix_hash[wiki]:
                if i[0] == j[0] and abs(i[1] - j[1]) <= size:
                    counter += 1
    return counter


def calc_ppmi(frequency_matrix, hash_matrix):
    amount_of_words = 0
    for i in frequency_matrix:
        amount_of_words += len(frequency_matrix[i])

    ppmi_matrix = frequency_matrix

    for row in range(1, len(ppmi_matrix)):
        for col in range(1, len(ppmi_matrix[row])):
            if ppmi_matrix[row][col] != 0:
                numerator = (2+ppmi_matrix[row][col])/(amount_of_words + 2 * len(ppmi_matrix[0]) * len(ppmi_matrix))
                denominator = (len(hash_matrix[ppmi_matrix[row][0]])/amount_of_words) * \
                              (len(hash_matrix[ppmi_matrix[0][col]])/amount_of_words)
                ppmi_matrix[row][col] = max(math.log2(numerator/denominator), 0)

    return ppmi_matrix


if __name__ == "__main__" :
    file = open("eng_wikipedia_2016_10K-sentences.txt", 'r+', encoding="utf8")
    text = data_fixed(file)
    matrix = create_matrix()
    matrix_hash = make_hash_table(text)

    fc2 = frequency_counts(matrix, matrix_hash, 2)
    fc5 = frequency_counts(matrix, matrix_hash, 5)

    ppmi2 = calc_ppmi(fc2, matrix_hash)
    ppmi5 = calc_ppmi(fc5, matrix_hash)
    print(ppmi5)







