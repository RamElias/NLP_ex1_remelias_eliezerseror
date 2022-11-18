import re

file = open("eng_wikipedia_2016_10K-sentences - Copy.txt", 'r+', encoding="utf8")


def data_fixed(file):
    text = file.readlines()
    marks_remove(text)
    add_tokens(text)


def marks_remove(text):
    for i in range(len(text)):
        arr = re.sub(r'[^\w\s]', '', text[i])
        text[i] = arr
        print(text[i])


def add_tokens(text):
    for i in range(len(text)):
        arr = text[i].split('\t')
        text[i] = arr[0] + "<S>{0}</S>\n".format((arr[1].split('\n'))[0])
        print(text[i])








