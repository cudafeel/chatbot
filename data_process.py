# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:25:57 2017

@author: zou
"""

import os
from collections import Counter
import jieba
import pickle

question_file = './data/question.txt'
answer_file = './data/answer.txt'


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()

    return data

def extract_cn_word(file, min_freq):
    words_count = Counter()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            seg_list = jieba.cut(line)
            for ch in seg_list:
                words_count[ch] = words_count[ch] + 1
                           
    sorted_list = [[v[1], v[0]] for v in words_count.items()]
    sorted_list.sort(reverse=True)
    
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    start_id = len(special_words)
    word_to_int = {}
    
    for i in range(start_id):
        word_to_int[special_words[i]] = i
    
    for index, item in enumerate(sorted_list):
        word = item[1]
        if item[0] < min_freq:
            break
        word_to_int[word] = start_id + index
#        id2word_dict[start_id+ index] = word
    return word_to_int

def save_dict_to_pkl(dic, word_to_int):
    output = open(dic, 'wb')
    pickle.dump(word_to_int, output)
    output.close()  

def load_dict_from_pkl(dic):
    pkl_file = open(dic, 'rb')
    word_to_int = pickle.load(pkl_file)
    return word_to_int

question_word_to_int_dict = './data/question.pkl'
answer_word_to_int_dict = './data/answer.pkl'

if not os.path.exists(question_word_to_int_dict):
    question_word_to_int = extract_cn_word(question_file, 1)
    save_dict_to_pkl(question_word_to_int_dict, question_word_to_int) 

if not os.path.exists(answer_word_to_int_dict):
    answer_word_to_int = extract_cn_word(answer_file, 1)
    save_dict_to_pkl(answer_word_to_int_dict, answer_word_to_int)     

question_word_to_int = load_dict_from_pkl(question_word_to_int_dict)
question_int_to_word = {word_i: word for word, word_i in question_word_to_int.items()}

answer_word_to_int = load_dict_from_pkl(answer_word_to_int_dict)
answer_int_to_word = {word_i: word for word, word_i in answer_word_to_int.items()}

source_letter_ids_path = './data/source_letter_ids.pkl'
target_letter_ids_path = './data/target_letter_ids.pkl'


if not os.path.exists(source_letter_ids_path):
    source_sentences = load_data(question_file)
    source_letter_ids = [[question_word_to_int.get(letter, question_word_to_int['<UNK>']) for letter in jieba.cut(line)] for line in source_sentences.split('\n')]
    save_dict_to_pkl(source_letter_ids_path, source_letter_ids) 

if not os.path.exists(target_letter_ids_path):
    target_sentences = load_data(answer_file)
    target_letter_ids = [[answer_word_to_int.get(letter, answer_word_to_int['<UNK>']) for letter in jieba.cut(line)] + [answer_word_to_int['<EOS>']] for line in target_sentences.split('\n')] 
    save_dict_to_pkl(target_letter_ids_path, target_letter_ids) 

source_letter_ids = load_dict_from_pkl(source_letter_ids_path)
target_letter_ids = load_dict_from_pkl(target_letter_ids_path)
