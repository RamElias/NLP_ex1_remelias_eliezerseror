import re
import nltk
import numpy as np


def data_fixed(file):
    text = file.readlines()

    for i in range(len(text)):
        sentence = "<S>"
        arr = text[i].split("\t")
        word_arr = arr[1].split(" ")

        for w in word_arr:
            print(w)
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
        matrix.append([i]+[""] * (len(cols)-1))

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
    for i in range(50, 20050):
        arr = cols[i].split('\t')
        cols_arr.append(arr[1])

    return cols_arr


def make_hash_table(text):
    hash_t = {}
    # file =  open('eng_wikipedia_2016_10K-sentences.txt', 'r', encoding="utf8")
    # text = file.readlines()

    for i, sentence in enumerate(text):
        sentence = sentence[sentence.index("<S>") + len("<S>"):sentence.index("</S>") - 1].split(" ")

        for j, word in enumerate(sentence):
            if word in hash_t:
                hash_t[word].append([i, j])
            else:
                hash_t[word] = [[i, j]]

    return hash_t


if __name__ == "__main__" :
    file = open("eng_wikipedia_2016_10K-sentences - Copy.txt", 'r+', encoding="utf8")
    text = data_fixed(file)
    print(text)
    matrix = create_matrix()
    matrix_hash = make_hash_table(text)
    print(matrix_hash)




