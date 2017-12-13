# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:25:57 2017

@author: zou
"""

import numpy as np
from data_process import load_dict_from_pkl
import jieba

from seq2seq_model import seq2seq_model, get_model_inputs
import tensorflow as tf

#from data_process import load_data
#question_file = './data/question.txt'
#answer_file = './data/answer.txt'
#source_sentences = load_data(question_file)
#target_sentences = load_data(answer_file)
#source_sentences[:60].split('\n')
#target_sentences[:60].split('\n')

question_word_to_int_dict = 'C:/Users/zou/Desktop/PythonCode/NLP/seq2seq_chat/data/question.pkl'
answer_word_to_int_dict = 'C:/Users/zou/Desktop/PythonCode/NLP/seq2seq_chat/data/answer.pkl'

# Build int2letter and letter2int dicts

question_word_to_int = load_dict_from_pkl(question_word_to_int_dict)
question_int_to_word = {word_i: word for word, word_i in question_word_to_int.items()}

answer_word_to_int = load_dict_from_pkl(answer_word_to_int_dict)
answer_int_to_word = {word_i: word for word, word_i in answer_word_to_int.items()}

# Convert characters to ids
source_letter_ids_path = './data/source_letter_ids.pkl'
target_letter_ids_path = './data/target_letter_ids.pkl'

source_letter_ids = load_dict_from_pkl(source_letter_ids_path)
target_letter_ids = load_dict_from_pkl(target_letter_ids_path)

#print("Example source sequence")
#print(source_letter_ids[:6])
#print([[question_int_to_word[i] for i in sentence] for sentence in source_letter_ids[:6]])
#print("\n")
#print("Example target sequence")
#print(target_letter_ids[:6])
#print([[answer_int_to_word[i] for i in sentence] for sentence in target_letter_ids[:6]])

epochs = 1000
batch_size = 2
rnn_size = 32
num_layers = 1

encoding_embedding_size = 64
decoding_embedding_size = 64

learning_rate = 0.001


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        
        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))
        
        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))
        
        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

# Split data to training and validation sets
train_source = source_letter_ids[batch_size:]
train_target = target_letter_ids[batch_size:]
valid_source = source_letter_ids[:batch_size]
valid_target = target_letter_ids[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           question_word_to_int['<PAD>'],
                           answer_word_to_int['<PAD>']))

display_step = 1 # Check training loss after every batches

model_file = "./model/chat_model.ckpt" 

def run_training(model_file):
    tf.reset_default_graph()
    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs(batch_size)
    end_points = seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  question_word_to_int, answer_word_to_int,
                  encoding_embedding_size, decoding_embedding_size, 
                  rnn_size, num_layers, batch_size, train=True)
    
    start_epoch = 0
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        try:
            for epoch_i in range(start_epoch+1, epochs+1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                        get_batches(train_target, train_source, batch_size,
                                   question_word_to_int['<PAD>'],
                                   answer_word_to_int['<PAD>'])):
                    
                    # Training step
                    _, loss = sess.run([
                            end_points['train_op'], 
                            end_points['loss']
                            ], feed_dict=
                            {input_data: sources_batch,
                             targets: targets_batch,
                             lr: learning_rate,
                             target_sequence_length: targets_lengths,
                             source_sequence_length: sources_lengths})
        
                    # Debug message updating us on the status of the training
                    if batch_i % display_step == 0:  
                        # Calculate validation cost
                        validation_loss = sess.run(
                        end_points['loss'],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})
                        
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      epochs, 
                                      batch_i, 
                                      len(train_source) // batch_size, 
                                      loss, 
                                      validation_loss))
                        # Save Model                                
                saver.save(sess, model_file)
                start_epoch += 1
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, model_file)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch_i+1))
    
def source_to_seq(text):
    '''Prepare the text for the model'''
    sequence_length = 20
    return [question_word_to_int.get(word, question_word_to_int['<UNK>']) for word in jieba.cut(text)]+ [question_word_to_int['<PAD>']]*(sequence_length-len(text))


def predict_sentence(input_sentence, model_file):
    text = source_to_seq(input_sentence)
    batch_size = 2
    
    tf.reset_default_graph()
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs(batch_size)
    end_points = seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  question_word_to_int, answer_word_to_int,
                  encoding_embedding_size, decoding_embedding_size, 
                  rnn_size, num_layers, batch_size, train=False)
    with tf.Session() as sess:
        # Load saved model
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_file)  
        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(end_points['inference_logits'], {input_data: [text]*batch_size, 
                                          target_sequence_length: [len(text)]*batch_size, 
                                          source_sequence_length: [len(text)]*batch_size})[0]
        
    pad = question_word_to_int["<PAD>"]
    
    print('Original Text:', input_sentence)
    
    print('\nSource')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([question_int_to_word[i] for i in text])))
    
    print('\nTarget')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([answer_int_to_word[i] for i in answer_logits if i != pad])))



run_training(model_file)

#input_sentence = '你还好吗?'
#predict_sentence(input_sentence, model_file)




