import torch
import os
import pickle
from model import BertLM
import consts
from tqdm import tqdm as progressbar
from pathlib import Path


class DataHandler():
    def __init__(self, file_path, data_name, model_type, layer=12, control=False,
                 ablation=False, language='', attribute='POS'):
        self.file_path = file_path
        if 'train' in str(file_path):
            self.set_name = 'train_'
        elif 'dev' in str(file_path):
            self.set_name = 'dev_'
        else:
            self.set_name = 'test_'
        self.data_name = data_name
        self.model_type = model_type
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clean_sentences = {}
        self.words_dict, self.poss_dict = {}, {}
        self.pos_to_idx, self.idx_to_pos = {}, {}
        self.layer = layer
        self.control = control
        # self.small = small_dataset
        # self.small_str = '_small' if small_dataset else ''
        self.ablation = ablation
        self.attribute = attribute

    def save_obj(self, obj, file_name):
        path = os.path.join('pickles', 'ablation' if self.ablation else '',
                            self.data_name, self.model_type,
                            self.device.type + '_' + self.set_name + file_name + '.pkl')
        with open(path, 'w+b') as f:
            pickle.dump(obj, f)

    def load_obj(self, file_name):
        path = os.path.join('pickles', 'ablation' if self.ablation else '', self.data_name, self.model_type,
                            self.device.type + '_' + self.set_name + file_name + '.pkl')
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def create_dicts(self):
        parsed_data_path = Path('pickles', 'UM', self.model_type, self.language, self.set_name + 'parsed.pkl')
        with open(parsed_data_path, 'rb') as f:
            self.parsed_data = pickle.load(f)
        sent_path = Path('pickles', 'UM', self.model_type, self.language, self.set_name + 'sentences.pkl')
        with open(sent_path, 'rb') as f:
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

    def get_features(self):
        if self.ablation:
            self.get_features_for_ablation()
        elif self.data_name == 'UM':
            return

    def get_features_for_ablation(self):
        self.features_by_sentence = self.load_obj('features_layer_' + str(self.layer))
        if self.features_by_sentence is not None:
            self.features_tensor = self.load_obj('features_tensor_' + str(self.layer))
            return
        if self.data_name == 'UM':
            dump_path = Path('pickles', self.data_name, self.model_type, self.language,
                             self.set_name + 'features_layer_' + str(self.layer))
            if dump_path.exists():
                return
        self.features_tensor, self.features_by_sentence = [], {}
        bert_model = BertLM(self.model_type, self.layer)
        total_loss, total_correct, total_tokens = 0., 0., 0.
        skipped = []
        for idx in progressbar(range(min(consts.ABLATION_NUM_SENTENCES, len(self.clean_sentences.keys())))):
            bert_res = bert_model(self.clean_sentences[idx])
            if bert_res is None:
                print('sentence idx: {}'.format(idx))
                skipped.append(idx)
                continue
            loss, correct_preds, tokens, sentence_features = bert_res
            self.features_tensor.append(sentence_features)
            self.features_by_sentence[idx] = sentence_features
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
                pickle.dump(self.features_by_sentence, f)
            if skipped:
                dump_path = Path('pickles', self.data_name, self.model_type,
                                 self.language, self.set_name + 'skipped_sentences.pkl')
                with open(dump_path, 'wb+'):
                    pickle.dump(skipped, f)
            return
        self.save_obj(self.features_by_sentence, 'features_layer_' + str(self.layer))
        self.save_obj(self.features_tensor, 'features_tensor_layer_' + str(self.layer))


class UMDataHandler(DataHandler):
    def __init__(self, file_path, data_name, model_type, layer=12, control=False,
                 language='', attribute='POS'):
        super(UMDataHandler, self).__init__(file_path, data_name=data_name, model_type=model_type, layer=layer,
                                            control=control,
                                            language=language, attribute=attribute)
        self.parsed_data = None

    def create_dataset(self, neurons: list = None):
        if neurons is None:
            neurons = list(range(consts.BERT_OUTPUT_DIM))
        att_path = Path('pickles', 'UM', self.model_type, self.language, self.attribute)
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
                                  'embedding': emb})

        label_to_idx_file = Path(att_path, 'label_to_idx.pkl')
        if not label_to_idx_file.exists():
            possible_labels = set([word[self.attribute] for word in filtered_data])
            label_to_idx = {label: i for i, label in enumerate(possible_labels)}
            with open(label_to_idx_file, 'wb+') as f:
                pickle.dump(label_to_idx, f)
        else:
            with open(label_to_idx_file, 'rb') as f:
                label_to_idx = pickle.load(f)
        assert len(label_to_idx) > 1
        control_labels_file = Path(att_path, self.set_name + 'control_labels')
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
                pickle.dump(control_labels, f)
        else:
            with open(control_labels_file, 'rb') as f:
                control_labels = pickle.load(f)
        if self.control:
            probing_data = [(word['embedding'], control_labels[word['word']]) for word in filtered_data]
        else:
            probing_data = [(word['embedding'], label_to_idx[word[self.attribute]])
                            for word in filtered_data]
        return probing_data
