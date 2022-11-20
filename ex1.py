import re


def marks_remove(text):
    for i in range(len(text)):
        arr = re.sub(r'[^\w\s]', '', text[i])
        text[i] = arr
        #print(text[i])


def add_tokens(text):
    for i in range(len(text)):
        arr = text[i].split('\t')
        text[i] = arr[0] + "<S>{0}</S>\n".format((arr[1].split('\n'))[0])
        #print(text[i])


def lower_case(text):
    for i in range(len(text)):
        arr = text[i].lower()
        text[i] = arr
        #print(text[i])


def replace_numbers(text):
    for i in range(len(text)):
        arr = text[i].split('\t')
        #rint(arr[0], "m")
        text[i] = arr[0] + '\t'+ re.sub(r'([0-9]+)', '<!DIGIT!>', arr[1])
        #print(text[i])


def fix_words(text):
    for i in range(len(text)):
        arry = text[i].split(" ")
        for j in range(len(arry)):
            if not check_word(arry[j]):
                arry[j].replace(arry[j],"<UNK>")
        print(text[i])


def check_word(word):
    for i in range(len(word)):
        if word[i] != (r'[a-zA-Z]'):
            return False
    return True





def data_fixed(file):
    text = file.readlines()
    marks_remove(text)
    lower_case(text)
    replace_numbers(text)
    fix_words(text)

    add_tokens(text)



if __name__ == "__main__" :
    file = open("eng_wikipedia_2016_10K-sentences.txt", 'r+', encoding="utf8")
    data_fixed(file)



