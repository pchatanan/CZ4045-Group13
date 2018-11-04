# detect comparative expressions
# examples of comparative words/phrases : less, lesser, greater, more
# may miss out on some words due to the nature of the tagging algorithm
import multiprocessing
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

def search_comparative(text):
    #data = json.loads(text)
    tokenized_sentences = sent_tokenize(text)
    comparative_words = []
    for sentence in tokenized_sentences:
        word_tokens = word_tokenize(sentence)
        tagged_words = pos_tag(word_tokens)
        for i in tagged_words:
            if i[1] == "JJR" or i[1] == "RBR":
                comparative_words.append(i[0])
    return comparative_words

def store_result(result):
    for i in result:
        if i not in comparative:
            comparative.append(i)
    
if __name__ == "__main__":
    comparative = []
    print("This application attempts to identify comparative words in your corpus.")
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z (Windows) followed by and Enter to register your input.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    print(contents)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    for i in contents:
        pool.apply_async(search_comparative, args=[i],callback=store_result)
    pool.close()
    pool.join()
    print("Comparative words found :", len(comparative))
    print("The words are : ", end="")
    for i in range(0, len(comparative)):
        if i != (len(comparative)-1):
            print(comparative[i]+", ", end="")
        else:
            print(comparative[i])
