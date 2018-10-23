# Noun Phrase Summarizer
## Top 20 noun phrases
import ast
import re
from nltk import word_tokenize, pos_tag
from collections import Counter
import multiprocessing
import matplotlib.pyplot as plt
import json
from data_analysis import plot_frequency




def find_noun_phrases(review):
    '''
    Returns list of noun phrases in the string
    : param review: Review string
    '''
    noun_phrases=[]
    regex = re.compile(r'(?:(?:\w+ DT )?(?:\w+ JJ )+)\w+ (?:N[NP]+|PRN)')
    pos_tags=pos_tag(word_tokenize(review))
    review=''.join([x+" "+y+" " for (x,y) in pos_tags])
    result=regex.findall(review)
    result=[x.replace(" JJ",'').replace(" NN",'').replace(" NP",'').replace(" DT",'').replace(" PRN",'') for x in result]
    return result

def log_noun_phrase_result(noun_phrases):
    noun_phrases_list.extend(noun_phrases)


data=[]
with open('/Users/Ajinkya/Downloads/CellPhoneReview.json',encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))
review_data=[item["reviewText"] for item in data]
noun_phrases_list=[]
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
for review in review_data:
    pool.apply_async(find_noun_phrases, args=[review], callback=log_noun_phrase_result)
pool.close()
pool.join()
noun_phrases_counter=Counter(noun_phrases_list)
top_20_noun_phrases=noun_phrases_counter.most_common(20)
plot_frequency(noun_phrases_counter,"top-20 noun phrases","noun phrases","no of times","v")

## Representative noun phrases for top 3 products
top_3_products=['B005SUHPO6','B0042FV2SI','B008OHNZI0']
for product in top_3_products:
    noun_phrases_list=[]
    review_data=[item["reviewText"] for item in data if item['asin']==product]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    for review in review_data:
        pool.apply_async(find_noun_phrases, args=[review], callback=log_noun_phrase_result)
    pool.close()
    pool.join()
    noun_phrases_product_counter=Counter(noun_phrases_list)
    noun_phrases_counter_dict=dict(noun_phrases_counter)
    noun_phrases_product_counter_dict=dict(noun_phrases_product_counter)
    for key in noun_phrases_product_counter_dict:
        val1=noun_phrases_product_counter_dict[key]
        val2=noun_phrases_counter_dict[key]
        noun_phrases_product_counter_dict[key]=2*val1+(val2-val1)
    print("\n\nProduct-->"+product)
    print(Counter(noun_phrases_product_counter_dict).most_common(10))
