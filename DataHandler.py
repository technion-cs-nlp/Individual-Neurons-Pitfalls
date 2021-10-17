from typing import List

import torch
import re
import os
import pickle
from model import BertWordEmbeds, BertLM
import consts
# from constants import BERT_OUTPUT_DIM
from tqdm import tqdm as progressbar
# from conllu import TokenList
from pathlib import Path

class DataHandler():
    def __init__(self, file_path: str, data_name, model_type, layer=12, control=False, small_dataset = False,
                 ablation=False, language = '', attribute='POS'):
        self.file_path=file_path
        if 'train' in str(file_path):
            self.set_name = 'train_'
        elif 'dev' in str(file_path):
            self.set_name = 'dev_'
        else:
            self.set_name = 'test_'
        # self.data_name = 'PENN TO UD' if 'PENN TO UD' in file_path else \
        #     'PENN' if 'PENN' in file_path else 'UD'
        self.data_name = data_name
        self.model_type = model_type
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clean_sentences = {}
        self.words_dict, self.poss_dict = {}, {}
        self.pos_to_idx, self.idx_to_pos = {}, {}
        # self.num_sentences = len(open(file_path,'r').readlines())
        self.layer = layer
        self.control = control
        self.small = small_dataset
        self.small_str = '_small' if small_dataset else ''
        self.ablation = ablation
        self.attribute = attribute

    def save_obj(self, obj, file_name):
        path = os.path.join('pickles', 'ablation' if self.ablation else '',
                            self.data_name, self.model_type, self.device.type +'_' + self.set_name + file_name + '.pkl')
        with open(path, 'w+b') as f:
            pickle.dump(obj,f)

    def load_obj(self, file_name):
        path = os.path.join('pickles', 'ablation' if self.ablation else '', self.data_name, self.model_type,
                             self.device.type +'_' + self.set_name + file_name + '.pkl')
        if not os.path.exists(path):
            return None
        with open(path,'rb') as f:
            return pickle.load(f)

    def create_dicts(self):
        if self.data_name == 'PENN':
            self.create_dicts_penn()
        elif self.data_name == 'UD':
            self.create_dicts_ud()
        elif self.data_name == 'PENN TO UD':
            self.create_dicts_penn_to_ud()
        elif self.data_name == 'UM':
            parsed_data_path = Path('pickles', 'UM', self.model_type, self.language, self.set_name + 'parsed.pkl')
            with open(parsed_data_path, 'rb') as f:
                self.parsed_data = pickle.load(f)
            sent_path = Path('pickles','UM', self.model_type, self.language, self.set_name + 'sentences.pkl')
            with open(sent_path,'rb') as f:
                self.clean_sentences = pickle.load(f)

    def count_values_for_att(self):
        histogram = {}
        for word in self.parsed_data:
            if not word['attributes'].get(self.attribute):
                continue
            if not histogram.get(word['attributes'][self.attribute]):
                histogram[word['attributes'][self.attribute]] = set()
            histogram[word['attributes'][self.attribute]].add(word['word'])
        return histogram

    def create_dicts_penn(self):
        self.clean_sentences = self.load_obj('clean_sentences')
        if self.clean_sentences != None:
            self.words_dict = self.load_obj('words_dict')
            self.poss_dict = self.load_obj('poss_dict')
            path = os.path.join('pickles', self.data_name, self.device.type + '_train_pos_to_idx.pkl')
            with open(path, 'rb') as f:
                self.pos_to_idx = pickle.load(f)
            self.control_tags = self.load_obj('control_tags')
        else:
            self.clean_sentences = {}
            with open(self.file_path,'r') as f:
                for idx,sentence in enumerate(f):
                    clean_sentence = re.sub(r"_[^\s]*",'',sentence[:-1])
                    self.clean_sentences[idx]=clean_sentence
                    words = clean_sentence.split()
                    self.words_dict[idx]=words
                    poss = re.sub(r"[^_^\s]*_",'',sentence[:-1]).split()
                    if idx == self.num_sentences-1:
                        poss.append('.')
                    self.poss_dict[idx]=poss
                self.save_obj(self.clean_sentences,'clean_sentences')
                self.save_obj(self.words_dict,'words_dict')
                self.save_obj(self.poss_dict,'poss_dict')
            if self.set_name == 'train':
                self.create_pos_idx_dicts()

    def create_pos_idx_dicts(self):
        if self.set_name == 'train_':
            self.pos_set = set(self.all_poss)
            for idx,pos in enumerate(self.pos_set):
                self.pos_to_idx[pos]=idx
                self.idx_to_pos[idx]=pos
            self.save_obj(self.pos_to_idx,'pos_to_idx')
            self.save_obj(self.idx_to_pos,'idx_to_pos')
        else:
            path = os.path.join('pickles', self.data_name, self.device.type + '_train_pos_to_idx.pkl')
            with open(path, 'rb') as f:
                self.pos_to_idx = pickle.load(f)
        self.all_poss = [self.pos_to_idx[pos] for pos in self.all_poss]
        self.save_obj(torch.tensor(self.all_poss), 'labels_tensor'+self.small_str)

    def get_features(self):
        if self.ablation:
            self.get_features_for_ablation()
        elif self.data_name == 'UM':
            return
        else:
            self.get_features_for_attribute()

    def get_features_for_attribute(self):
        self.features_by_sentece = self.load_obj('features_layer_'+str(self.layer)+self.small_str)
        if self.features_by_sentece != None:
            self.features_tensor = self.load_obj('features_tensor_'+str(self.layer)+self.small_str)
            return
        self.features_tensor, self.features_by_sentece = [], {}
        bert_model = BertWordEmbeds(self.layer)
        for idx in progressbar(self.clean_sentences.keys()):
            sentence_features = bert_model(self.clean_sentences[idx])
            self.features_tensor.append(sentence_features)
            self.features_by_sentece[idx] = sentence_features
        self.features_tensor = torch.cat(self.features_tensor)
        self.save_obj(self.features_by_sentece,'features_layer_'+str(self.layer)+self.small_str)
        self.save_obj(self.features_tensor,'features_tensor_layer_'+str(self.layer)+self.small_str)

    def get_features_for_ablation(self):
        self.features_by_sentece = self.load_obj('features_layer_' + str(self.layer) + self.small_str)
        if self.features_by_sentece != None:
            self.features_tensor = self.load_obj('features_tensor_' + str(self.layer) + self.small_str)
            return
        if self.data_name == 'UM':
            dump_path = Path('pickles', self.data_name, self.model_type, self.language, self.set_name+'features_layer_'+str(self.layer))
            if dump_path.exists():
                return
        self.features_tensor, self.features_by_sentece = [], {}
        bert_model = BertLM(self.model_type, self.layer)
        total_loss, total_correct, total_tokens = 0., 0., 0.
        skipped = []
        for idx in progressbar(range(min(consts.ABLATION_NUM_SENTENCES, len(self.clean_sentences.keys())))):
            bert_res = bert_model(self.clean_sentences[idx])
            if bert_res == None:
                print('sentence idx: {}'.format(idx))
                skipped.append(idx)
                continue
            loss, correct_preds, tokens, sentence_features = bert_res
            self.features_tensor.append(sentence_features)
            self.features_by_sentece[idx] = sentence_features
            total_loss += loss
            total_correct += correct_preds
            total_tokens += tokens
            if total_tokens > consts.ABLATION_NUM_TOKENS:
                break
        self.features_tensor = torch.cat(self.features_tensor)
        print('total tokens: {}'.format(total_tokens))
        print('whole vector loss: ', total_loss)
        print('whole vector accuracy: ', total_correct / total_tokens)
        if self.data_name == 'UM':
            dump_path = Path('pickles', self.data_name, self.model_type, self.language,
                             self.set_name + 'features_layer_' + str(self.layer))
            with open(dump_path, 'wb+') as f:
                pickle.dump(self.features_by_sentece, f)
            if skipped:
                dump_path = Path('pickles', self.data_name, self.model_type,
                                 self.language, self.set_name + 'skipped_sentences.pkl')
                with open(dump_path,'wb+'):
                    pickle.dump(skipped, f)
            return
        self.save_obj(self.features_by_sentece, 'features_layer_'+str(self.layer)+self.small_str)
        self.save_obj(self.features_tensor, 'features_tensor_layer_'+str(self.layer)+self.small_str)

    def create_dicts_penn_to_ud(self):
        self.clean_sentences = self.load_obj('clean_sentences' + self.small_str)
        self.words_dict = self.load_obj('words_dict' + self.small_str)
        path = os.path.join('pickles', self.data_name, self.device.type + '_train_pos_to_idx.pkl')
        with open(path, 'rb') as f:
            self.pos_to_idx = pickle.load(f)
        self.poss_dict = self.load_obj('poss_dict' + self.small_str)
        if self.poss_dict == None:
            penn_poss_dict = self.load_obj('poss_dict_penn_form')
            self.poss_dict = self.convert_penn_tags_to_ud_tags(penn_poss_dict)
            self.all_poss = [self.pos_to_idx[tag] for i in range(len(self.poss_dict))
                             for tag in self.poss_dict[i]]
            self.all_poss = torch.tensor(self.all_poss)
            self.save_obj(self.poss_dict, 'poss_dict' + self.small_str)
            self.save_obj(self.all_poss, 'labels_tensor' + self.small_str)
        else:
            self.all_poss = self.load_obj('labels_tensor'+self.small_str)

    def convert_penn_tags_to_ud_tags(self, penn_poss_dict):
        mapping = consts.penn_to_ud_labels
        ud_poss_dict = {}
        for i, sentence in penn_poss_dict.items():
            ud_poss_dict[i] = [mapping[word] for word in sentence]
        return ud_poss_dict


    def create_dicts_ud(self):
        self.clean_sentences = self.load_obj('clean_sentences'+self.small_str)
        if self.clean_sentences != None:
            self.words_dict = self.load_obj('words_dict'+self.small_str)
            self.poss_dict = self.load_obj('poss_dict'+self.small_str)
            path = os.path.join('pickles', self.data_name, self.device.type + '_train_pos_to_idx.pkl')
            with open(path, 'rb') as f:
                self.pos_to_idx = pickle.load(f)
            self.all_poss = self.load_obj('labels_tensor'+self.small_str)
            self.control_tags = self.load_obj('control_tags'+self.small_str)
        else:
            self.clean_sentences = {}
            self.all_words, self.all_poss = [], []
            self.num_sentences = 0
            with open(self.file_path,'r') as f:
                clean_sentence = ''
                words, tags = [], []
                for idx, line in enumerate(f):
                    if line.startswith('# text'):
                        clean_sentence = line[len('# text = '):]
                        self.clean_sentences[self.num_sentences] = clean_sentence
                        continue
                    elif line.startswith('#'):
                        continue
                    elif line == '\n' and words:
                        self.words_dict[self.num_sentences] = words
                        self.poss_dict[self.num_sentences] = tags
                        self.all_words += words
                        self.all_poss += tags
                        self.clean_sentences[self.num_sentences] = ' '.join(words)
                        words, tags = [], []
                        self.num_sentences += 1
                        if self.small and self.num_sentences >= consts.SMALL_DATA_SIZE[self.set_name]:
                            break
                        continue
                    #normal line
                    words.append(line.split()[1])
                    tags.append(line.split()[3])
                self.save_obj(self.clean_sentences, 'clean_sentences'+self.small_str)
                self.save_obj(self.words_dict, 'words_dict'+self.small_str)
                self.save_obj(self.poss_dict, 'poss_dict'+self.small_str)
            self.create_pos_idx_dicts()


class SingleNeuron(DataHandler):
    def __init__(self, file_path:str):
        super(SingleNeuron, self).__init__(file_path)
    def create_dataset(self):
        feature_and_label = {k: [] for k in range(consts.BERT_OUTPUT_DIM)}
        for i in range(self.num_sentences):
            if i % 500 == 0:
                print('sentence ', i)
            for j in range(len(self.poss_dict[i])):
                for k in range(consts.BERT_OUTPUT_DIM):
                    feature = self.features_by_sentece[i][j][k]
                    word_label = self.pos_to_idx[self.poss_dict[i][j]]
                    feature_and_label[k].append((feature,word_label))
        return feature_and_label

    def create_dataset_for_neuron(self, neuron:int):
        features_and_labels = []
        for i in range(self.num_sentences):
            for j in range(len(self.poss_dict[i])):
                feature = self.features_by_sentece[i][j][neuron]
                word_label = self.pos_to_idx[self.poss_dict[i][j]]
                features_and_labels.append((feature,word_label))
        return features_and_labels

class SingleLabel(DataHandler):
    def __init__(self, file_path:str):
        super(SingleLabel, self).__init__(file_path)
    def create_dataset_for_neuron_and_label(self, neuron:int, label:int):
        features_and_labels = []
        for i in range(self.num_sentences):
            for j in range(len(self.poss_dict[i])):
                feature = self.features_by_sentece[i][j][neuron]
                word_label = 1 if self.poss_dict[i][j] == label else 0
                features_and_labels.append((feature, word_label))
        return features_and_labels

class UMDataHandler(DataHandler):
    def __init__(self, file_path:str, data_name:str, model_type, layer=12, control=False, small_dataset=False,
                 language='', attribute='POS'):
        super(UMDataHandler, self).__init__(file_path, data_name=data_name, model_type=model_type, layer=layer, control=control,
                                         small_dataset=small_dataset, language=language, attribute=attribute)
        self.parsed_data = None

    def create_dataset(self, neurons: list=None):
        if neurons == None:
            neurons = list(range(consts.BERT_OUTPUT_DIM))
        att_path = Path('pickles','UM', self.model_type, self.language,self.attribute)
        with open(Path(att_path, 'values_to_ignore.pkl'), 'rb') as f:
            values_to_ignore = pickle.load(f)
        filtered_data = []
        for word in self.parsed_data:
            if not word['attributes'].get(self.attribute):
                continue
            if word['attributes'][self.attribute] in values_to_ignore:
                continue
            id = word['word']
            att = word['attributes'][self.attribute]
            emb = torch.tensor(word['embedding'][self.layer][neurons])
            emb = emb.to(self.device)
            filtered_data.append({'word': id,
                                  self.attribute: att,
                                  'embedding':emb})

        label_to_idx_file = Path(att_path, 'label_to_idx.pkl')
        if not label_to_idx_file.exists():
            possible_labels = set([word[self.attribute] for word in filtered_data])
            label_to_idx = {label: i for i, label in enumerate(possible_labels)}
            with open(label_to_idx_file,'wb+') as f:
                pickle.dump(label_to_idx,f)
        else:
            with open(label_to_idx_file,'rb') as f:
                label_to_idx = pickle.load(f)
        assert len(label_to_idx) > 1
        control_labels_file = Path(att_path, self.set_name+'control_labels')
        if not control_labels_file.exists():
            possible_labels = len(set([word[self.attribute] for word in filtered_data]))
            words_set = {word['word'] for word in filtered_data}
            if self.set_name == 'train_':
                control_labels = {word: torch.randint(high=possible_labels, size=[1]).item() for word in words_set}
            else:
                train_control_labels_file = Path(att_path, 'train_control_labels')
                with open(train_control_labels_file, 'rb') as f:
                    train_control_labels = pickle.load(f)
                control_labels = {word: train_control_labels[word] if word in train_control_labels else
                                torch.randint(high=possible_labels, size=[1]).item()
                                  for word in words_set}
            with open(control_labels_file, 'wb+') as f:
                pickle.dump(control_labels,f)
        else:
            with open(control_labels_file, 'rb') as f:
                control_labels = pickle.load(f)
        if self.control:
            probing_data = [(word['embedding'], control_labels[word['word']])for word in filtered_data]
        else:
            probing_data = [(word['embedding'], label_to_idx[word[self.attribute]])
                            for word in filtered_data]
        return probing_data

class DataSubset(DataHandler):
    def __init__(self, file_path:str, data_name:str, layer=12, control=False, small_dataset=False,
                 language='', attribute='POS'):
        super(DataSubset, self).__init__(file_path, data_name=data_name, layer=layer, control=control,
                                         small_dataset=small_dataset, language=language, attribute=attribute)
    def create_dataset(self, neurons: list = None):
        if not neurons:
            neurons = list(range(consts.BERT_OUTPUT_DIM))
        features_and_labels=[]
        tag_idx = 0
        for i in range(len(self.clean_sentences)):
            for j in range(len(self.poss_dict[i])):
                features=self.features_by_sentece[i][j][neurons]
                if self.control:
                    word_label = self.control_tags[tag_idx]
                    tag_idx += 1
                else:
                    word_label = self.pos_to_idx[self.poss_dict[i][j]]
                features_and_labels.append((features,word_label))
        return features_and_labels