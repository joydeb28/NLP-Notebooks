# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Aug  6 19:40:26 2018

@author: joy
"""
import json
abs_path1 = "benchmarking_data/Train/"

abs_path2 = "benchmarking_data/Validate/"
import re
reg = re.compile('[A-z]*\_([A-z]*)\_[A-z]*')
reg2 = re.compile('[A-z]*\_([A-z]*)')


def make_data_for_intent_from_json(json_file,txt_file):
    
    json_d = json.load(open(abs_path1+json_file))
    json_dict = json_d[reg.match(json_file).group(1)]
    
    wr = open("Intent_Data/"+txt_file,'w')
    
    for i in json_dict:
        each_list = i['data']
        sent =""
        for i in each_list:
            sent = sent + i['text']+ " "
        sent =sent[:-1]
        for i in range(3):
            sent = sent.replace("  "," ")
        wr.write(sent)
        wr.write('\n')
        print(sent)        
                    

def make_data_from_json(json_file,txt_file):
    
    json_d = json.load(open(abs_path2+json_file))
    json_dict = json_d[reg2.match(json_file).group(1)]
    
    wr = open(abs_path2+txt_file,'w')
    
    for i in json_dict:
        each_list = i['data']
        for i in each_list:
            try:
                words = i['text'].split()
                print(words[0]+' '+'B-'+i['entity'])
                wr.write(words[0]+' '+'B-'+i['entity'])
                wr.write('\n')
                for word in words[1:]:
                    print(word+' '+'I-'+i['entity'])
                    wr.write(word+' '+'I-'+i['entity'])
                    wr.write('\n')
                #print(i['text']+'\t'+i['entity'])

            except:
                words = i['text'].split()
                for word in words:
                    print(word+' '+'O')
                    wr.write(word+' '+'O')
                    wr.write('\n')
        print('\n')
        wr.write('\n')
        

def make_data_from_json_train(json_file,txt_file):
    
    json_d = json.load(open(abs_path1+json_file))
    json_dict = json_d[reg.match(json_file).group(1)]
    
    wr = open(abs_path1+txt_file,'w')
    
    for i in json_dict:
        each_list = i['data']
        for i in each_list:
            try:
                words = i['text'].split()
                print(words[0]+' '+'B-'+i['entity'])
                wr.write(words[0]+' '+'B-'+i['entity'])
                wr.write('\n')
                for word in words[1:]:
                    print(word+' '+'I-'+i['entity'])
                    wr.write(word+' '+'I-'+i['entity'])
                    wr.write('\n')
                #print(i['text']+'\t'+i['entity'])

            except:
                words = i['text'].split()
                for word in words:
                    print(word+' '+'O')
                    wr.write(word+' '+'O')
                    wr.write('\n')
        print('\n')
        wr.write('\n')

import nltk
def make_data_from_json_train_pos(json_file,txt_file):
    
    json_d = json.load(open(abs_path2+json_file))
    json_dict = json_d[reg2.match(json_file).group(1)]
    
    wr = open(abs_path2+txt_file,'w')
    
    for i in json_dict:
        each_list = i['data']
        sent = ""
        for i in each_list:
            sent = sent+i['text']+" "
            sent = sent.replace("  "," ")
        if sent[-1]==" ":
            sent = sent[:-1]
        words = []
        pos_tags = nltk.pos_tag(sent.split())
        print(pos_tags,sent)
        pos_tag_dict = {j:k for j,k in pos_tags}
        for i in each_list:
            try:
                
                words = i['text'].split()
                print(words[0]+' '+pos_tag_dict[words[0]]+" "+'B-'+i['entity'])
                wr.write(words[0]+" "+pos_tag_dict[words[0]]+" "+'B-'+i['entity'])
                wr.write('\n')
                for word in words[1:]:
                    print(word+' '+pos_tag_dict[word]+" "+'I-'+i['entity'])
                    wr.write(word+' '+pos_tag_dict[word]+" "+'I-'+i['entity'])
                    wr.write('\n')
                #print(i['text']+'\t'+i['entity'])

            except:
                words = i['text'].split()
                for word in words:
                    print(word+' '+pos_tag_dict[word]+" "+'O')
                    wr.write(word+' '+pos_tag_dict[word]+" "+'O')
                    wr.write('\n')
        print('\n')
        wr.write('\n')

        
import re   
import json
import os     
def make_data_from_snips(input_path):
    
    for r,d,f in os.walk(input_path):
 
        for filename  in f:
            label = os.path.basename(r)
            source = os.path.join(r,filename)
            

            
            if os.path.splitext(filename)[-1] != '.txt':
                continue
            
            
            
            
            read_file = open(source)
    

            pattern = re.compile(r'(?:[[])(?P<value>.*?)(?:[]])(?:[(])(?P<name>.*?)(?:[)])')
    
            corpus = dict()
            corpus[label] = list()
            for i in read_file:
                data = list()
                
                it = pattern.finditer(i)
                
                sent_len = len(i.strip())
                
                if sent_len == 0:
                    continue

                last_span = 0
                for m in it:
                    
                    head = i[last_span:m.span()[0]]
                    obj = dict()
                    if head.strip():
                        obj['text'] = head
                    
                        data.append(obj)
                    
                    obj = dict()
                    obj['text'] = m.group('value')
                    obj['entity'] = m.group('name')
                    
                    data.append(obj)
                    
                    last_span = m.span()[1]
                if last_span:
                    obj = dict()
                    if i[last_span :].strip():
                        obj['text'] = i[last_span :]
                        data.append(obj)
                
                if data:
                    
                    corpus[label].append({'data': data})
            
            with open(os.path.join(r,filename.split()[0] + '.json'),'w',encoding='utf-8') as fp:
                json.dump(corpus,fp)
                
    
    

    
#make_data("book_restaurant_train.csv","book_restaurant_train.txt")
'''
make_data_from_json_train_pos("train_AddToPlaylist_full.json","train_AddToPlaylist_full.txt")
make_data_from_json_train_pos("train_BookRestaurant_full.json","train_BookRestaurant_full.txt")
make_data_from_json_train_pos("train_GetWeather_full.json","train_GetWeather_full.txt")
make_data_from_json_train_pos("train_PlayMusic_full.json","train_PlayMusic_full.txt")
make_data_from_json_train_pos("train_RateBook_full.json","train_RateBook_full.txt")
make_data_from_json_train_pos("train_SearchCreativeWork_full.json","train_SearchCreativeWork_full.txt")
make_data_from_json_train_pos("train_SearchScreeningEvent_full.json","train_SearchScreeningEvent_full.txt")
'''

make_data_from_json_train_pos("validate_AddToPlaylist.json","validate_AddToPlaylist.txt")
make_data_from_json_train_pos("validate_BookRestaurant.json","validate_BookRestaurant.txt")
make_data_from_json_train_pos("validate_GetWeather.json","validate_GetWeather.txt")
make_data_from_json_train_pos("validate_PlayMusic.json","validate_PlayMusic.txt")
make_data_from_json_train_pos("validate_RateBook.json","validate_RateBook.txt")
make_data_from_json_train_pos("validate_SearchCreativeWork.json","validate_SearchCreativeWork.txt")
make_data_from_json_train_pos("validate_SearchScreeningEvent.json","validate_SearchScreeningEvent.txt")



#make_data_from_snips("flight_data")
