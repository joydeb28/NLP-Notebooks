#!/usr/bin/env python
# coding: utf-8


import os
from pickle import dump
import pandas as pd
from numpy import array
from numpy.random import shuffle
from keras.utils.vis_utils import plot_model
from keras.layers import Input ,LSTM, Embedding, Dense, TimeDistributed

from keras.layers import RepeatVector, Bidirectional, Dropout

from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint

#from generate_phoneme import get_phoneme_list
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

from numpy import argmax
from keras.models import load_model
import time
from pickle import load
from generate_phoneme import get_phoneme


def get_phoneme_list():
    phoneme_list = ['start','end','<pad>', '<unk>', 'SP', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 
                        'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 
                        'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 
                        'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 
                        'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 
                        'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 
                        'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
    return  phoneme_list

class PreProcessing():
    def __init__(self,dataset=None):
        self.pairs = None
        self.dataset = dataset
        self.training_data = None
        self.testing_data = None
        
    def get_tokenizer(self):
        tokenizer = Tokenizer()
        phoneme_list = get_phoneme_list()
        tokenizer.fit_on_texts(phoneme_list)
        return tokenizer
        
    def max_length(self,lines):
        return max(len(line.split()) for line in lines)

    def get_length(self):
        correct_data = self.dataset[:, 0]
        faulty_data = self.dataset[:, 1]
        tar_data_length = max(len(line.split()) for line in correct_data)
        src_data_length = max(len(line.split()) for line in faulty_data)
        return tar_data_length,src_data_length
    
    def encode_sequences(self,tokenizer, length, lines):
        X = tokenizer.texts_to_sequences(lines)
        X = pad_sequences(X, maxlen=length, padding='post')
        return X
 
    def encode_output(self,sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = np.array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y
    
    
    
    
    
class LoadData():
    def __init__(self,file):
        self.file = file
        self.data_frame = pd.read_csv(self.file,sep=",")
        self.pairs = None
        self.dataset = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.train_data_length = None
        self.test_data_length = None
        
    def load_data(self):
        self.data_frame = pd.read_csv(self.file)
        self.pairs = self.data_frame.values.tolist()
        wrong_ids = list()
        
        for id_ll in range(len(self.pairs)):
            self.pairs[id_ll][0] = "start " + self.pairs[id_ll][0] + " end"
            for id_l in range(len(self.pairs[id_ll])):
                try:
                    if '   ' in self.pairs[id_ll][id_l]:
                        #print(self.pairs[id])
                        self.pairs[id_ll][id_l] = self.pairs[id_ll][id_l].replace('.','')
                        self.pairs[id_ll][id_l] = self.pairs[id_ll][id_l].replace('   ',' SP ')
                        
                except:
                    print("ee:",self.pairs[id_ll])
                    wrong_ids.append(id_ll)
                    print(id_ll,id_l)
        print("Wrong ids: ",wrong_ids)       
        
        for id in wrong_ids:
            del self.pairs[id]
        #print(self.pairs)
        self.dataset = array(self.pairs)
        shuffle(self.dataset)
        self.train_data_length = int(len(self.pairs)*0.9)
        self.test_data_length = len(self.pairs) - self.train_data_length


class PhonemeModel():
    
    def __init__(self,model_structure,X_train,Y_train,X_test,Y_test,epochs,batch_size):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_structure = model_structure
    
    def get_tokens(self):
        token_words = get_phoneme_list()
        token_index = dict([(word, i+1) for i, word in enumerate(token_words)])
        
        reverse_token_index = dict((i, word) for word, i in token_index.items())

        return token_index,reverse_token_index
    
    def generate_batch(self, X, y, max_length_src, max_length_tar,num_decoder_tokens, batch_size ):
        token_index,_ = self.get_tokens()
        ''' Generate a batch of data '''
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        try:
                            encoder_input_data[i, t] = token_index[word]
                        except Exception as e:
                            print("Wrong word:",word)
                            print("Exception:",e)
                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = token_index[word]
                        if t>0:
                            decoder_target_data[i, t - 1, token_index[word]] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)
                
    def enc_dec_model(self, vocab_size, src_data_length, tar_data_length, n_units):
        
        current_directory = os.getcwd()
        #data_folder = os.path.join(current_directory,"..","data")
        model_folder = os.path.join(current_directory,"..","models")
        
        #encoder
        encoder_input = Input(shape = (None,))
        encoder_emb =  Embedding(vocab_size, n_units, mask_zero = True)(encoder_input)
        encoder_lstm = LSTM(n_units,return_state = True)
        encoder_outputs,encode_h,encoder_c = encoder_lstm(encoder_emb)
        encoder_states = [encode_h,encoder_c]
        
        #decoder
        decoder_input = Input(shape = (None,))
        decoder_emb_layer = Embedding(vocab_size+1, n_units, mask_zero = True)
        decoder_emb = decoder_emb_layer(decoder_input)
        decoder_lstm = LSTM(n_units,return_sequences=True,return_state = True)
        decoder_out,decode_h,decoder_c = decoder_lstm(decoder_emb,initial_state = encoder_states)
        decoder_dense = Dense(vocab_size,activation="softmax")
        decoder_out = decoder_dense(decoder_out)
        self.model = Model([encoder_input,decoder_input],decoder_out)
        #compile
        self.model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=['accuracy'])

        #fit
        tar_data_length,src_data_length = tar_data_length,src_data_length
        train_gen = self.generate_batch(X_train,Y_train,src_data_length,tar_data_length,vocab_size,self.batch_size)
        test_gen = self.generate_batch(X_test,Y_test,src_data_length,tar_data_length,vocab_size,self.batch_size)
        train_samples_steps = len(X_train) / self.batch_size
        val_samples_steps = len(X_test) / self.batch_size
        
        cat = model_structure["cat"]
        version = model_structure["version"]
        
        
        filename = os.path.join(model_folder,cat+'_Enc_Dec_base_model-'+version+'.h5')
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks=[checkpoint]
        
        self.model.fit_generator(generator = train_gen,
                    steps_per_epoch = train_samples_steps,
                    epochs=self.epochs,
                    validation_data = test_gen,
                    validation_steps = val_samples_steps,callbacks = callbacks)
        
        #eocoder setup
        
        self.encoder_model = Model(encoder_input, encoder_states)
        
        # Decoder setup
        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        dec_emb2 = decoder_emb_layer(decoder_input)
        
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary
        
        # Final decoder model
        self.decoder_model = Model([decoder_input] + decoder_states_inputs,[decoder_outputs2] + decoder_states2)
        
        
        model_folder = model_structure["model_folder"]
        
        
        encoder_model_filename = os.path.join(model_folder,cat+'_Encoder_base_model-v'+version+'.h5')
        dump(self.encoder_model, open(encoder_model_filename, 'wb'))
        decoder_model_filename = os.path.join(model_folder,cat+'_Decoder_base_model-v'+version+'.h5')
        dump(self.decoder_model, open(decoder_model_filename, 'wb'))
        
class Prediction():
    
    def __init__(self,model_structure,src_data_length,tar_data_length,data_vocab_size):
        self.model_structure = model_structure
        self.src_data_length = src_data_length
        self.tar_data_length = tar_data_length
        self.data_vocab_size = data_vocab_size
        cat = self.model_structure["cat"]
        version = self.model_structure["version"]
        current_directory = os.getcwd()
        #data_folder = os.path.join(current_directory,"..","data")
        model_folder = os.path.join(current_directory,"..","models")
        encoder_model_filename = os.path.join(model_folder,cat+'_Encoder_base_model-v'+version+'.h5')
        self.encoder_model = load(open(encoder_model_filename, 'rb'))
        decoder_model_filename = os.path.join(model_folder,cat+'_Decoder_base_model-v'+version+'.h5')
        self.decoder_model = load(open(decoder_model_filename, 'rb'))
    
    def get_tokens(self):
        token_words = get_phoneme_list()
        token_index = dict([(word, i+1) for i, word in enumerate(token_words)])
        
        reverse_token_index = dict((i, word) for word, i in token_index.items())

        return token_index,reverse_token_index
    
    def generate_batch(self, x, y, max_length_src, max_length_tar, num_decoder_tokens, batch_size ):
        token_index,_ = self.get_tokens()
        ''' Generate a batch of data '''
        while True:
            for j in range(0, len(x), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(x[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        try:
                            encoder_input_data[i, t] = token_index[word]
                        except Exception as e:
                            print("Wrong word:",word)
                            print("Exception:",e)
                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = token_index[word]
                        if t>0:
                            decoder_target_data[i, t - 1, token_index[word]] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)
                
    
    
    def decode_sequence(self,input_seq):
        target_token_index,reverse_target_char_index = self.get_tokens()
        #print("target_token_index:",target_token_index)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = target_token_index["start"]
        stop_condition = False
        decoded_phoneme = []
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_phoneme.append(sampled_char)
    

            if len(decoded_phoneme)> 26 or sampled_char=="end":
                del decoded_phoneme[-1]
                stop_condition = True
    
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
    
            states_value = [h, c]
    
        return decoded_phoneme
    
    def get_preiction(self,item,actual_result):
        
        phoneme_item = get_phoneme(item)
        actual_phoneme = get_phoneme(actual_result)
        
        for id in range(len(phoneme_item)):
            if phoneme_item[id] == " ":
                phoneme_item[id] = "SP"
                
        for id in range(len(actual_phoneme)):
            if actual_phoneme[id] == " ":
                actual_phoneme[id] = "SP"
                
        phoneme_str = " ".join(phoneme_item)
        actual_str = " ".join(actual_phoneme)
        
        dataset = [[phoneme_str,actual_str]]
        dataset = array(dataset)
        x = dataset[:1,1]
        y = dataset[:1,0]
        test_gen = self.generate_batch(x, y, self.src_data_length, self.tar_data_length, self.data_vocab_size, 1)
        (input_seq, actual_output), _ = next(test_gen)
        enoded_phoneme_data = self.decode_sequence(input_seq)
        
        return enoded_phoneme_data
        
        
if __name__ == '__main__':
    
    model_structure = {"cat" :"area","version" : "1","model_type" : "enc_dec"}
    epochs = 100
    batch_size = 35#ld.train_data_length
    
    
    cat = model_structure["cat"]
    version = model_structure["version"]
    model_type = model_structure["model_type"]
    current_directory = os.getcwd()
    data_folder = os.path.join(current_directory,"..","data")
    model_folder = os.path.join(current_directory,"..","models")
    model_structure["model_folder"] = model_folder
    # Loading Data
    #ld = LoadData("data/input_test.csv")
    output_data_file = os.path.join(data_folder,cat+"_training_phoneme_data.csv")
    ld = LoadData(output_data_file)
    ld.load_data()
    preprocess_obj = PreProcessing(ld.dataset)
    # prepare Correct data tokenizer
    data_tokenizer = preprocess_obj.get_tokenizer()    
    data_vocab_size = len(data_tokenizer.word_index) + 1
    
    # prepare tokenizer
    tar_data_length,src_data_length = preprocess_obj.get_length()#max_length(dataset[:, 0])
    print('Vocabulary Size: %d' % data_vocab_size)
    print('Correct Length: %d' % (tar_data_length))
    
    #src_data_length = preprocess_obj.max_length(dataset[:, 1])
    print('Actual Length: %d' % (src_data_length))
    dataset = ld.dataset
    X_train = dataset[:ld.train_data_length,1]
    Y_train = dataset[:ld.train_data_length,0]
    X_test = dataset[:ld.test_data_length,1]
    Y_test = dataset[:ld.test_data_length,0]
    
    #model
    
    is_traning_requried = input("Do you want train your model?(y/n) : ")
    
    if is_traning_requried.lower()=="y":
        
        md = PhonemeModel(model_structure,X_train,Y_train,X_test,Y_test,epochs,batch_size)
        
        
        md.enc_dec_model(data_vocab_size, src_data_length, tar_data_length, 512)
        
        model = md.model
        encoder_model = md.encoder_model
        decoder_model = md.decoder_model
    
    
    
    #prediction
    text_dict = load(open(os.path.join("..","models",cat+"_phoneme_to_text_dict.pkl"),'rb'))
    pred = Prediction(model_structure,src_data_length,tar_data_length,data_vocab_size)
    test_file = os.path.join(data_folder,cat+"_test_cases.txt")
    fread = open(test_file,'r')
    test_cases_list = fread.read().splitlines()
    test_list = list()
    for item in test_cases_list:
        test_list.append(item.split(','))
    
    for each_list in test_list:
        actual_result = each_list[0]
        for item in each_list[1:]:
            start_time = time.time()
            print("Input:",item)
            print("target:",actual_result)
            result = pred.get_preiction(item,actual_result)
            print("Predicted Phoneme:",result)
            result = " ".join(result)
            result_list = result.split(" SP ")
            
            pred_text = ""
            for item in result_list:
                try:
                    if pred_text:
                        pred_text = pred_text +" "+ text_dict[item]
                    else:
                        pred_text = text_dict[item]
                except:
                    pass            
            print("Time:",(time.time()-start_time)*1000," ms" )
            print("Predicted Text: ",pred_text)
            print("Actual Text: ",actual_result)
            if pred_text.lower() == actual_result.lower():
                print("Pass")
            else:
                print("Fail")
            print("\n")
    
    