import pandas as pd
import json
import re
import ast
import pickle
import glob
from time import process_time, sleep
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
import networkx as nx
from networkx.algorithms import community
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from pyvis.network import Network
import string
import colorama
from colorama import Fore
import termtables as tt
string.punctuation
import openai
import time
import os
import seaborn as sns
import itertools
import openai

import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df=pd.read_csv('./data/d_cls_fishbone.csv',index_col=0)
with open('./buffer/fish-bone-selection.txt', 'r', encoding='utf-8') as file:
    lines_list = file.readlines()
lines_list = [line.strip() for line in lines_list]

lines_list[0]=lines_list[0].replace('Task: ','')
lines_list[1]=lines_list[1].replace('Issue (please select less than 3 issues in that task): ','')

matching_rows = df[df['corpusid'] == lines_list[0][0]]

target_issue=lines_list[1].split(', ')
target_issue=target_issue[0:3]

# task_corpusid
task_corpusid=[]
for i in range(len(df)):
    if df['class'][i]==int(lines_list[0][0]):
        task_corpusid.append(df['corpusid'][i])
        
d_insight=pd.read_csv('./data/Insight_dataset_included_summary-concise.csv',index_col=0)

matching_rows = d_insight[d_insight['corpusid'] == task_corpusid[0]]
for i in range(len(task_corpusid)):
    matching_rows_add = d_insight[d_insight['corpusid'] == task_corpusid[i]]
    matching_rows=pd.concat([matching_rows, matching_rows_add], ignore_index=True, axis=0)


columns = ['sourceNodeId','targetNodeId','cite_info','relationshipType','flag']
d_edges=pd.DataFrame(columns=columns)



count=0
for i in range(len(matching_rows)):
    try:
        List=eval(matching_rows['ref_paper_id'][i])
        # cite_dic=eval(Insight_dataset['cite_text'][i])
        count=count+1
    except:
        continue
    for j in List:
        if j in list(matching_rows['corpusid']):
            # d_edges_append=pd.DataFrame(data=[[Insight_dataset['corpusid'][i],j,str(List),cite_dic[j].replace('\n',''),'cites',count]], columns=columns)
            d_edges_append=pd.DataFrame(data=[[matching_rows['corpusid'][i],j,str(List),'cites',count]], columns=columns)
            d_edges=pd.concat([d_edges, d_edges_append], ignore_index=True, axis=0)
            
            
# total_id=list(set(d_edges['sourceNodeId']))
# total_id.extend(list(set(d_edges['targetNodeId'])))

total_id=list(set(matching_rows['corpusid']))


#clear output noise
def remove_before_colon(text):
    return re.sub(r'^.*?:', '', text)


#id -> index -> content
re_num={}
for i in range (len(total_id)):
    content={}
    content['num']=i
    content['type']=1
    for j in range(len(matching_rows)):
        if matching_rows['corpusid'][j]==total_id[i]:
            if type(matching_rows['paper-title'][j])==float:
                continue
            # content['author-date']=total[total_id[i]]
            content['title']=matching_rows['paper-title'][j]
            content['pdfurl']=matching_rows['pdfurl'][j]
            content['conclusion']=remove_before_colon(matching_rows['Summary'][j])
            content['content']=matching_rows['Content'][j]
            content['ReSolved']=remove_before_colon(matching_rows['ReSolved'][j])
            content['Finding']=remove_before_colon(matching_rows['Finding'][j])

            if 'a survey' in content['title'] or 'A survey' in content['title'] or 'a Survey' in content['title'] or 'A Survey' in content['title']:
                content['type']=0
            else:
                content['type']=1
            
            # print('true')
            break
        
    re_num[total_id[i]]=content
    
list_source=[]
list_target=[]
for i in range(len(d_edges)):
    if d_edges['sourceNodeId'][i]==d_edges['targetNodeId'][i]:
        continue
   
    list_source.append(re_num[d_edges['sourceNodeId'][i]]['num'])
    list_target.append(re_num[d_edges['targetNodeId'][i]]['num'])
    
    
#type revise
for i in re_num:
    if type(re_num[i]['ReSolved'])==float:
        re_num[i]['ReSolved']=''
    if type(re_num[i]['Finding'])==float:
        re_num[i]['Finding']=''
        
#nltk_word

def word_div(strings):
    # Convert the string to lowercase
    strings = strings.lower()
    
    # Remove punctuation
    strings = strings.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the string into words
    token = nltk.word_tokenize(strings)
    
    # Get stopwords
    stop_words = stopwords.words('english')
    
    # Filter out stopwords
    tokens = [word for word in token if word not in stop_words]
    
    # Define specific words to filter out
    filter_word = ['from', 'subject', 're', 'edu', 'use', 'introduction', 'conclusion', 'evaluation', 'figure', 'table',
                   'survey', 'et', 'al', 'research', 'we', 'work', 'issue', 'also', 'future', 'however', 'finally',
                   'task', 'tasks', 'paper', 'nlp', 'recent', 'natural', 'language', 'processing', 'nevertheless','result','show','aim','method','results']
    
    # Perform POS tagging to filter nouns (NN), verbs (VB), and adjectives (JJ)
    pos_tagged = nltk.pos_tag(tokens)
    
    clean_token = []
    for word, pos in pos_tagged:
        # Check if the word is not in the filter list and is not a number
        if word not in filter_word and re.search(r'^\d+$', word) is None:
            # Only keep nouns (NN), verbs (VB), and adjectives (JJ)
            if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
                # Use lemmatizer to reduce words to their base form (noun category)
                clean_token.append(lemmatizer.lemmatize(word, pos="n"))
    
    return clean_token


#Similarity

lemmatizer = WordNetLemmatizer()
st = LancasterStemmer()
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
sentences=[]
strings_total=''
branch=2
pair_list=[]
pair_list_total=[]
for item in re_num:
    if 'content' not in re_num[item].keys():
        # print(re_num[item]['num'])
        continue
    
    sentences.append(re_num[item]['content'])
    strings_total=strings_total+re_num[item]['content']
    



#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i_1 in range(len(cosine_scores)-1):
    for j_1 in range(i_1+1, len(cosine_scores)):
        pairs.append({'index': [i_1, j_1], 'score': cosine_scores[i_1][j_1]})

#Sort scores in decreasing order
pairs_dec = sorted(pairs, key=lambda x: x['score'], reverse=True)

#Simliarity of each article (n-branch)
start=0
for i in range(1,len(pairs)):
    strings_pair=''
    flag=pairs[i-1]['index'][0]
    i_1, j_1 = pairs[i]['index']
    if i_1!=flag:
        pair=pairs[start:i]
        pair = sorted(pair, key=lambda x: x['score'], reverse=True)
        # for building tree
        for j in range(branch):
            pair[j]['s1']=sentences[pair[j]['index'][0]]
            pair[j]['s2']=sentences[pair[j]['index'][1]]
            strings_pair=sentences[pair[j]['index'][0]]+sentences[pair[j]['index'][1]]
            # print(Fore.BLACK + "{} \t\t {} \t\t Score: {:.4f}".format(sentences[pair[j]['index'][0]], sentences[pair[j]['index'][1]], pair[j]['score']))

            #nltk->上位語,top:10
            clean_token=word_div(strings_pair)
            freq_dist = nltk.FreqDist(clean_token)
            # print(Fore.BLUE + 'hypernym:',freq_dist.most_common(10))
            pair[j]['hypernym']=freq_dist.most_common(10)

            #nltk ->共通語
            S_0=word_div(sentences[pair[j]['index'][0]])
            S_1=word_div(sentences[pair[j]['index'][1]])
            and_list=set(S_0)&set(S_1)
            # print(Fore.MAGENTA + 'common_word:',and_list)
            pair[j]['common_word']=and_list

            #add data to list
            pair_list.append(pair[j])

        start=i

for j in range(len(pairs)):
    pairs[j]['s1']=sentences[pairs[j]['index'][0]]
    pairs[j]['s2']=sentences[pairs[j]['index'][1]]
    strings_pair=sentences[pairs[j]['index'][0]]+sentences[pairs[j]['index'][1]]
    # print(Fore.BLACK + "{} \t\t {} \t\t Score: {:.4f}".format(sentences[pair[j]['index'][0]], sentences[pair[j]['index'][1]], pair[j]['score']))

    #nltk->Hypernym,top:10
    clean_token=word_div(strings_pair)
    freq_dist = nltk.FreqDist(clean_token)
    # print(Fore.BLUE + 'hypernym:',freq_dist.most_common(10))
    pairs[j]['hypernym']=freq_dist.most_common(10)

    #Common_word
    S_0=word_div(sentences[pairs[j]['index'][0]])
    S_1=word_div(sentences[pairs[j]['index'][1]])
    and_list=set(S_0)&set(S_1)
    # print(Fore.MAGENTA + 'common_word:',and_list)
    pairs[j]['common_word']=and_list



# arrange
FW_list=[]
Solved_list=[]
sentences_fw=[]
sentences_solved=[]
for key in re_num:
    if re_num[key]['Finding']!='' and type(re_num[key]['Finding'])!=float:
        FW_list.append(re_num[key]['num'])
        sentences_fw.append(re_num[key]['Finding'])
    if re_num[key]['ReSolved']!='' and type(re_num[key]['ReSolved'])!=float:
        Solved_list.append(re_num[key]['num'])
        sentences_solved.append(re_num[key]['ReSolved'])

#Compute embeddings
embeddings_FW = model.encode(sentences_fw, convert_to_tensor=True)
embeddings_Solved = model.encode(sentences_solved, convert_to_tensor=True)
cosine_scores_novelty = util.cos_sim(embeddings_FW, embeddings_Solved)



#node selection
s_c=5 #papers on one tree

avg_list=[]
max_list_index=[]

for i in range(len(cosine_scores_novelty)):
    l=[]
    for z in range(s_c):
        l.append(Solved_list[int(np.argsort(cosine_scores_novelty[i])[-(z+s_c)])])

    avg_list.append(sum(cosine_scores_novelty[i])/len(cosine_scores_novelty[i]))
    max_list_index.append(l)

avg_list_index_org=list(np.argsort(avg_list)[::-1])


#adjustment
avg_list_index=[]
for i in range(len(avg_list_index_org)):
    avg_list_index.append(FW_list[avg_list_index_org[i]])
    

# sim with selected issue from fish-bone
sim_issue_max=[]
for item in target_issue:
    sim_issue={}
    for key in re_num:
        # print(re_num[key]['content'])
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([item, re_num[key]['content']])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        print("Cosine Similarity:", cos_sim[0][0])
        sim_issue[re_num[key]['num']]=cos_sim[0][0]
    
    sim_issue_max.append(max(sim_issue, key=sim_issue.get))


#node-select-tree
root_max=len(target_issue) # init 
layer_max=5
leaf_lim=2 # number of leaf
tree_list=[] # node be selected
node_his=[]
for n in range(root_max):# Topn cited-root
     # history
    root_list=[] # root list   
    print(f'path{n}:')
    path_dict={}
    i=1
    avg_list=avg_list_index.copy()
    # print('rank:',i)
    # root=int(avg_list_index[n])
    root=sim_issue_max[n]
    print(root)
    
    node_his.append(root)
    root_list.append(root)

    while True:
        if i>layer_max:
            break
        if root_list==[]:
            break
        # if ancestors_num.index(sorted(ancestors_num,reverse=True)[0])==0:
        #     break
        print('Current_node:',root_list[0])
        # G_ext.nodes[root]['label']=f'layer{i}'
        
    
        for l in range(leaf_lim):
            for s in range(s_c):
                candidate=max_list_index[FW_list.index(root)][s]
                if candidate not in node_his:
                    path_dict.setdefault(root_list[0],[]).append(candidate)
                    print('candidate:',candidate)
                    root_list.append(candidate)
                    node_his.append(candidate)
                    break

        i=i+1
        root_list.pop(0)
        print('root_list:',root_list)
                
    tree_list.append(path_dict)
    
    
strings_list=[]
for i in range(len(tree_list)):
    strings=''
    nodes=list(set(tree_list[i]))
    for node in nodes:
        for item in list(re_num.values()):
            if node == item['num']:
                key = [k for k, v in re_num.items() if v['num'] == node][0]
                break
        strings=strings+re_num[key]['content']
    
    strings_list.append(strings)

color_list = [
    "Lavender",
    "Peach Puff",
    "Pale Goldenrod",
    "Light Pink",
    "Light Sky Blue",
    "Misty Rose",
    "Honeydew",
    "Thistle",
    "Light Yellow",
    "Light Coral",
]


import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


from nltk import stem
def imp_word(d1,d2):
    words_list=[]
    document1=tb(d1)
    document2=tb(d2)
    bloblist = [document1, document2]

    stemmer = stem.LancasterStemmer()
    stop_words = stopwords.words('english')
    
    for i, blob in enumerate(bloblist):
        # print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:7]:
            if word in stop_words:
                continue

            clean_word=stemmer.stem(word)
            # print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            words_list.append(str(word))
        break

    return words_list
    
def wrap_text(text, max_line_length):
    words = text.split()
    wrapped_lines = []
    current_line = []

    for word in words:
        # 如果当前行的长度加上新单词长度超过最大长度，则换行
        if len(' '.join(current_line + [word])) > max_line_length:
            wrapped_lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    
    # 把最后一行加入结果
    if current_line:
        wrapped_lines.append(' '.join(current_line))
    
    return '\n'.join(wrapped_lines)


#check edge
def check_node_has_edges(net, node_id):
    for edge in net.edges:
        if edge['from'] == node_id or edge['to'] == node_id:
            return True
    return False


# build-graph : Tree structure
inv=100
inv_title=40
# color_list=['Pink','Red','Blue','Yellow','Brown','Pink','Green']
G_Novelty= Network(notebook=True, 
cdn_resources='in_line',
directed=True,
height='600px',
width='100%',
font_color="blue",
# heading='Relevance-tree'
)
history=[]

# task_node
for i in range(len(target_issue)):
    G_Novelty.add_node(-(i+1),label=wrap_text(target_issue[i],inv_title), shape='eclipse', font={"size": 40},color='Thistle')


for i in range(len(tree_list)):
    color=color_list[i]
    flag=0
    for j in tree_list[i]:
        for item in list(re_num.values()):
            if j == item['num']:
                key_j = [k for k, v in re_num.items() if v['num'] == j][0]
                words_list=imp_word(re_num[key_j]['content'],strings_list[i])
                # try:
                G_Novelty.add_node(j,label = wrap_text(re_num[key_j]['title'],inv_title),
                title=str(key_j)+ '\n\n'+'- Unique Keyword: '+', '.join(words_list)+'\n\n'+'- Summary: '+ wrap_text(re_num[key_j]['conclusion'],inv)+'\n\n'+ re_num[key_j]['pdfurl'],
                shape='box',
                size=60,
                font={"size": 20},
                color=color,
                group=i
                )
                # except:
                #     G_Novelty.add_node(j, color=color,label=str(j),title='None',group=i)
                
                node_id_to_check = -(i+1)
                if check_node_has_edges(G_Novelty, node_id_to_check):
                    print('Has already connected to an edge')
                else:
                    G_Novelty.add_edge(-(i+1), j,
                        gravity=-8000, 
                        spring_length=10000,
                        spring_strength=0.001,
                        overlap=1,
                        )
                    
            
        if flag==0:
            # G_Novelty.nodes[G_Novelty.node_ids.index(j)]['shape']='ellipse'
            flag=1
            
        for z in tree_list[i][j]:
            key_z = [k for k, v in re_num.items() if v['num'] == z][0]
            words_list=imp_word(re_num[key_z]['content'],strings_list[i])
            # try:
            G_Novelty.add_node(z,label=wrap_text(re_num[key_z]['title'],inv_title),
            title=str(key_z) +'\n\n'+'- Unique Keyword: '+', '.join(words_list)+'\n\n'+
            '- Summary: '+ wrap_text(re_num[key_z]['conclusion'],inv)+'\n\n'+ re_num[key_z]['pdfurl'],
            shape='box',
            size=40,
            font={"size": 15},
            color=color,
            group=i)
            # except:
            #     G_Novelty.add_node(z, color=color,label=str(z),title='None',group=i)

            if [j, z] in history:
                continue

            # add edge(thickness,length)
            flag=0
            ## find corpus id
            cite_r='False'
            for c in range(len(d_edges)):
                if d_edges['targetNodeId'][c] == key_j and d_edges['sourceNodeId'][c] == key_z:
                    cite_r='True'
                    break
                if d_edges['sourceNodeId'][c] == key_j and d_edges['targetNodeId'][c] == key_z:
                    cite_r='True'
                    break
              

            for p in pairs:
                if p['index']==[j,z] or p['index']==[z,j]:
                    G_Novelty.add_edge(j, z,
                    gravity=-8000, 
                    label='\n\n'.join(list(p['common_word'])[0:10]),
                    title='Cite_relationship: '+cite_r + '\n\n'+
            '- ReSolved: '+ wrap_text(re_num[key_z]['ReSolved'],inv)+'\n\n'+ 
            '- Finding: '+ wrap_text(re_num[key_j]['Finding'],inv),
                    spring_length=10000,
                    width=float(p['score'])*5,
                    spring_strength=0.001,
                    overlap=1,
                    font={"size": 15},
                    )
                    flag=1
                    break
            # if flag==0:
            #     G_generate.add_edge(j, z, color=color,width=0.1)

            # G_generate.add_edge(j, z,
            # label='edge',
            # gravity=-10,
            # central_gravity=0.3,
            # spring_length=z,
            # spring_strength=0.001,
            # overlap=1)
            
            history.append([j,z])
     

G_Novelty.bgcolor="#ffffee"
G_Novelty.barnes_hut()

options = """const options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "levelSeparation": 320,
      "nodeSpacing": 155,
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0,
      "nodeDistance": 180,
      "damping": 0.46,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  }
}
"""

G_Novelty.set_options(options)
# G_generate.set_edge_smooth('dynamic')

# G_generate.show_buttons(filter_=['layout','physics'])
# G_generate.show_buttons()
G_Novelty.show('Relevance_tree.html',notebook=False)




