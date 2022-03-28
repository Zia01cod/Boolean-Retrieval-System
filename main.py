from nltk.stem import *
import pickle
import re
import os
import nltk as nltk
import fnmatch

path = os.path.dirname(os.path.abspath(__file__))
Num_docs = 42

def get_doc_list(s, dictionary):                # Get list of document IDs for a term from inverted matrix(dictionary)
    if 'AND' in s:
        return intersect([st.strip() for st in s.split('AND')], dictionary)
    if 'NOT' in s:
        return complement_list(dictionary[s.replace('NOT ', '').lower()])
    filtered = fnmatch.filter(dictionary.keys(), s.lower())
    res = []
    for i in filtered:
        res = res + dictionary[i]
    return res


def intersect_list(lst1, lst2):       # A intersection B
    res = []
    i = j = 0
    while i != len(lst1) and j != len(lst2):
        if lst1[i] == lst2[j]:
            res.append(lst1[i])
            i += 1
            j += 1
        elif lst1[i] < lst2[j]:
            i += 1
        else:
            j += 1

    return res


def union_list(lst1, lst2):         # A U B

    return sorted(lst1 + lst2)


def complement_list(lst):               # B' = U - B

    return list(set(range(Num_docs)) - set(lst))


def intersect(words, dictionary):           # for AND
    if len(words) == 1:
        return get_doc_list(words[0])
    res = get_doc_list(words[0], dictionary)
    words = words[1:]
    while len(words) != 0:
        res = intersect_list(res, get_doc_list(words[0], dictionary))
        words = words[1:]

    return res


def union(words, dictionary):           # for OR
    
    if len(words) == 1:
        return get_doc_list(words[0], dictionary)
    res = get_doc_list(words[0], dictionary)
    words = words[1:]
    while len(words) != 0:
        res = union_list(res, get_doc_list(words[0], dictionary))
        words = words[1:]

    return res


##########################################################################################################

def stem_words(tokens):
    stemmer = PorterStemmer()                               # inbuilt stemming function from nltk
    stemmed_words = [stemmer.stem(token) for token in tokens]
    return stemmed_words


def remove_stopwords(lst):
    res = []
    listl = []
    with open('stopwords.txt', 'r') as f:           # Getting stopwords from stopwords.txt
        for line in f:
            listl.append(line.strip())
    for i in lst:
        if i not in listl:
            # print(i)
            res.append(i)
    return res


def read_data(path):
    contents = []
    d_dict = {}
    i = 1
    for filename in os.listdir(path + "/shakespeare"):
        data = open(path + '/shakespeare/' + filename, 'r').read()
        contents.append((i, data))
        d_dict[i] = filename                                # key: doc_id ; value: file_name
        i += 1
    with open("document_id.pkl", "wb") as file:
        pickle.dump(d_dict, file)
    return contents


def get_all_words(data):
    tokens = []
    for token in data.values():
        tokens = tokens + token
    fdist = nltk.FreqDist(tokens)
    return list(fdist.keys())


def preprocess(contents):
    dataDict = {}
    for id, content in contents:
        res = re.findall(r"[\w']+", content)       # Removing whitespaces and specials except '
        for i in range(len(res)):
            res[i] = res[i].lower()                # Converting into lowercase
        print("Without Stopword removal: ","")
        print(res[:10])
        res = remove_stopwords(res)                 # Removing stopwords
        print("With Stopword Removal: ", "")
        print(res[:10])
        print("Without Stemming: ", "")
        print(res[:10])
        res = stem_words(res)                       # Stemming
        print("With stemming: ", "")
        print(res[:10])
        dataDict[id] = res
    return dataDict


def generate_inverted_index(data):
    all_words = get_all_words(data)     # Extracting token list from data dictionary
    index = {}
    for word in all_words:
        for doc, tokens in data.items():
            if word in tokens:
                if word in index.keys():
                    print(word + " is an Existing Word. Therefore, simply appending doc id")
                    index[word].append(doc)     #Existing Word
                else:
                    print(word + " is a new word. Expanding the dictionary.")
                    index[word] = [doc]         # New word
    with open("inverted_matrix.pkl", "wb") as file:
        pickle.dump(index, file)


#####################################################################################################


if __name__ == '__main__':

    # data = read_data(path)   # Getting raw data in form of dict {file_number: " file content", ....}
    # preprocessed_data = preprocess(data)   # Preprocessing the data before putting into inverted matrix
    # generate_inverted_index(preprocessed_data)     # Generating inverted Matrix

    with open("document_id.pkl", "rb") as in_file:
        doc_dictionary = pickle.load(in_file)
    print(len(doc_dictionary))
    with open("inverted_matrix.pkl", "rb") as in_file:
        dictionary = pickle.load(in_file)
    for i in dictionary.keys():
        print(i)
    while True:
        inp = input("Input boolean query: ")
        try:
            res = union([s.strip() for s in inp.split('OR')], dictionary)
            print("Results of " + inp + ":")
            n = 1
            for r in res:
                print(str(n) + ") ", end = " ")
                print(doc_dictionary[r])
                n+=1
            print()
        except KeyError:
            print("Not a valid boolean query!!!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
