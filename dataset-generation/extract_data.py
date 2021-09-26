import pandas as pd
import numpy as np
import os
import sys

data_dir = './'

print("------extracting deterministic data------")
obstag = pd.read_csv(data_dir + 'final_data/before_rerank_id/train/obstag.csv')
rcttag = pd.read_csv(data_dir + 'final_data/before_rerank_id/train/rcttag.csv')
movie_real_tag_list = pd.read_csv(data_dir + 
    'generate_data/movie_real_tag_list.csv', index_col='movieid')
movie_real_tag_list['taglist'] = movie_real_tag_list['taglist'].apply(eval)


movie_tag_dict = {}
user_tag_dict = {}

extract_data = pd.DataFrame(columns=['userid', 'tagid', 'islike'])

for i in rcttag.index:
    uid = rcttag.loc[i]['userid']
    mid = rcttag.loc[i]['movieid']
    tag = rcttag.loc[i]['tagid']
    mtag_list = movie_real_tag_list.loc[mid]['taglist']
    for mtag in mtag_list:
        if not (uid, mtag) in user_tag_dict:
            user_tag_dict[(uid, mtag)] = 0
    if not tag == -1:
        user_tag_dict[(uid, tag)] = 1

for i in obstag.index:
    uid = obstag.loc[i]['userid']
    mid = obstag.loc[i]['movieid']
    tag = obstag.loc[i]['tagid']
    if tag == -1:
        mtag_list = movie_real_tag_list.loc[mid]['taglist']
        for mtag in mtag_list:
            if not (uid, mtag) in user_tag_dict:
                user_tag_dict[(uid, mtag)] = 0
    else:
        user_tag_dict[(uid, tag)] = 1


for (uid, tag) in user_tag_dict.keys():
    extract_data = extract_data.append({'userid': uid,
                          'tagid': tag, 'islike': user_tag_dict[(uid, tag)]}, ignore_index=True)

extract_data.to_csv(data_dir + 'generate_data/extract.csv', header=True, index=False)
print("------end------")