import sentencepiece
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast, XLMRobertaTokenizer,\
    XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
import consts


class BertLM(nn.Module):
    def __init__(self, model_type, layer):
        super(BertLM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') if model_type == 'bert'\
            else XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased') if model_type == 'bert'\
            else XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')
        self.layer = layer

    def forward(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        labels = self.tokenizer(sentence, return_tensors="pt")['input_ids']
        if labels.shape[-1] > 512:  # BERT max sentence len
            print('sentence too long: {} tokens'.format(labels.shape[-1]))
            return None
        labels[0][0] = labels[0][-1] = -100  # ignore on loss computation
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels, output_hidden_states=True)
        loss = outputs[0]
        correct_preds = (outputs[1][0].argmax(dim=1)[1:-1] == labels[0][1:-1]).sum().item()
        features = outputs[2][self.layer].squeeze(0)
        return loss, correct_preds, labels.shape[1]-2, features


def specific_words_acc(relevant_indices, preds, labels, with_att:bool):
    if with_att:
        relevant_indices_tensor = torch.zeros_like(labels)
    else:
        relevant_indices_tensor = torch.where(labels == -100,
                                              torch.zeros_like(labels), torch.ones_like(labels))
    for sentence_index, relevant_words in enumerate(relevant_indices):
        for relevant_word in relevant_words:
            if with_att:
                relevant_indices_tensor[sentence_index][relevant_word] = 1
            else:
                relevant_indices_tensor[sentence_index][relevant_word] = 0
    gold_relevant_words = labels[relevant_indices_tensor > 0]
    num_relevant = gold_relevant_words.shape[0]
    correct_relevant = (preds[relevant_indices_tensor > 0] == gold_relevant_words).sum().item()
    return correct_relevant, num_relevant



class BertFromMiddle(nn.Module):
    def __init__(self, model_type, layer):
        super(BertFromMiddle, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased') if model_type == 'bert'\
            else XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        self.layer = layer
        self.bert = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased').to(self.device) if model_type == 'bert'\
            else XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').to(self.device)
        self.layers = self.bert.bert.encoder.layer[self.layer:] if model_type == 'bert' else self.bert.roberta.encoder.layer[self.layer:]
        self.classifier = self.bert.cls if model_type == 'bert' else self.bert.lm_head
        self.prefix_char = '##' if model_type == 'bert' else '\u2581'
        self.pad_token = 0 if model_type == 'bert' else 1
        self.special_tokens = [0, 101, 102] if model_type == 'bert' else [0, 1, 2]
        self.model_type = model_type

    def map_words_to_tokens(self, sentences, batch_labels):
        words_to_tokens = []
        for i in range(len(sentences)):
            words_to_tokens.append([])
            splits = 0
            for token_idx, word in enumerate(batch_labels.encodings[i].word_ids):
                if word is None:
                    continue
                curr_word_first = batch_labels.encodings[i].offsets[token_idx][0]
                prev_word_last = batch_labels.encodings[i].offsets[token_idx - 1][1]
                curr_token = batch_labels.encodings[i].tokens[token_idx]
                if curr_word_first == prev_word_last != 0:
                    # if self.model_type == 'bert' and not curr_token.startswith(self.prefix_char):
                    #     splits += 1
                    # elif self.model_type == 'xlm' and curr_token.startswith(self.prefix_char):
                    #     splits += 1
                    if bool(self.model_type == 'bert') ^ bool(curr_token.startswith(self.prefix_char)):
                        splits += 1
                if len(words_to_tokens[i]) == word - splits:
                    words_to_tokens[i].append([token_idx])
                else:
                    words_to_tokens[i][word - splits].append(token_idx)
        return words_to_tokens

    def map_tokens_to_words(self, words_to_tokens):
        tokens_to_words = []
        for sent_idx, sent in enumerate(words_to_tokens):
            tokens_to_words.append([])
            for word_idx, word in enumerate(sent):
                for token in word:
                    tokens_to_words[sent_idx].append(word_idx)
        return tokens_to_words

    def map_input_to_output(self, true_labels, preds, input_tokens_to_words):
        input_to_output = []
        for sent_idx in range(true_labels.shape[0]):
            input_to_output.append([])
            pred_sentence_ids = preds[sent_idx][true_labels[sent_idx]>0]
            pred_sentence_tokens = self.tokenizer.convert_ids_to_tokens(pred_sentence_ids)
            output_word_idx = -1
            for token_idx, token in enumerate(pred_sentence_tokens):
                # if self.model_type == 'bert' and not token.startswith('##'):
                if not bool(self.model_type == 'bert') ^ bool(token.startswith(self.prefix_char)):
                    output_word_idx += 1
                curr_token_input_word = input_tokens_to_words[sent_idx][token_idx]
                if len(input_to_output[sent_idx]) == curr_token_input_word:
                    input_to_output[sent_idx].append({output_word_idx})
                else:
                    input_to_output[sent_idx][curr_token_input_word].add(output_word_idx)
        return input_to_output


    def forward(self, sentences, features, subtokens_with_attribute=None):
        with torch.no_grad():
            batch_labels = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
            words_to_tokens = self.map_words_to_tokens(sentences, batch_labels)
            tokens_to_words = self.map_tokens_to_words(words_to_tokens)
            labels = batch_labels['input_ids']
            attention_mask = torch.where(labels == self.pad_token, torch.zeros_like(labels), torch.ones_like(labels))
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            padded_features = torch.zeros(list(labels.shape) + [consts.BERT_OUTPUT_DIM]).to(self.device)
            for i in range(len(features)):
                padded_features[i, :features[i].shape[0], :] = features[i]
            for layer in self.layers:
                padded_features = layer(padded_features, attention_mask=extended_attention_mask)[0]
            pred_scores = self.classifier(padded_features)
            transpose = []
            for sentence_scores in pred_scores:
                transpose.append(sentence_scores.T)
            pred_scores = torch.stack(transpose)
            preds = pred_scores.argmax(dim=1)
            for i in self.special_tokens:
                labels = torch.where(labels == i, torch.ones_like(labels) * (-100), labels)
            loss_func = nn.CrossEntropyLoss()
            res = dict.fromkeys(['loss', 'correct_all', 'num_all', 'correct_relevant', 'num_relevant',
                                 'correct_irrelevant', 'num_irrelevant'],None)
            res['loss'] = loss_func(pred_scores, labels).item()
            res['correct_all'] = (preds == labels).sum().item()
            res['num_all'] = (labels != -100).sum().item()
            res['correct_relevant'], res['num_relevant'] = \
                specific_words_acc(subtokens_with_attribute, preds, labels, True)
            res['correct_irrelevant'], res['num_irrelevant'] = \
                specific_words_acc(subtokens_with_attribute, preds, labels, False)
            res['pred_tokens'] = [[''.join(self.tokenizer.decode(preds[i][word]).split()) for word in words_to_tokens[i]] for i in range(len(sentences))]
        return res


class PosTagger(nn.Module):
    def __init__(self):
        super(PosTagger, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PosTaggerWholeVector(PosTagger):
    def __init__(self, num_labels, first_layer_size=0):
        super(PosTaggerWholeVector, self).__init__()
        self.fc1 = nn.Linear(consts.BERT_OUTPUT_DIM, num_labels).to(self.device)
    def forward(self, features):
        preds = self.fc1(features)
        return preds


class PosTaggerSubset(PosTagger):
    def __init__(self, first_layer_size, num_labels):
        super(PosTaggerSubset, self).__init__()
        self.fc1 = nn.Linear(first_layer_size, num_labels)
    def forward(self, features):
        preds = self.fc1(features)
        return preds
