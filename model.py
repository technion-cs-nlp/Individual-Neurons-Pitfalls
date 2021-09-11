import sentencepiece
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast, XLMRobertaTokenizerFast, XLMRobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
import consts

def subword_tokenize(tokenizer: BertTokenizer, tokens):
    """
    Returns: List of subword tokens, List of indices mapping each subword token to one real token.
    """
    indexed_subtokens = {}
    for i, sentence_tokens in enumerate(tokens):
        subtokens = [tokenizer.tokenize(t) for t in sentence_tokens.split()]
        indexed_subtokens[i] = []
        for idx, subtoks in enumerate(subtokens):
            for subtok in subtoks:
                indexed_subtokens[i].append((idx, subtok))

    return indexed_subtokens

class BertWordEmbeds(nn.Module):
    def __init__(self,layer=12):
        super(BertWordEmbeds, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',do_basic_tokenize=False)
        self.BertModel = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.device)
        self.BertModel.eval()
        self.layer=layer

    def forward(self, sentences):
        sentence_tokens = torch.tensor(self.tokenizer.encode(sentences)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sentence_features = self.BertModel(sentence_tokens,output_hidden_states=True)[2][self.layer][0]
        sentence_features = sentence_features[1:-1]
        wordpiece_sentence = self.tokenizer.tokenize(sentences)
        chunk, new_features = [], []
        idx = 0
        while idx < len(wordpiece_sentence):
            if wordpiece_sentence[idx].startswith('##') and not wordpiece_sentence[idx].endswith('##'):
                chunk.append(idx-1)
                while idx < len(wordpiece_sentence) and wordpiece_sentence[idx].startswith('##'):
                    chunk.append(idx)
                    idx+=1
                new_features.pop()
                mean_word_tensor = sentence_features[chunk].mean(axis=0)
                new_features.append(mean_word_tensor.unsqueeze(0))
                chunk = []
            if idx < len(wordpiece_sentence):
                new_features.append(sentence_features[idx].unsqueeze(0))
            idx+=1
        new_features = torch.cat(new_features).to(self.device)
        return new_features

class BertLM(nn.Module):
    def __init__(self, model_type, layer):
        super(BertLM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') if model_type == 'bert'\
            else XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased') if model_type == 'bert'\
            else XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')
        self.layer = layer

    def forward(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        labels = self.tokenizer(sentence, return_tensors="pt")['input_ids']
        if labels.shape[-1] > 512: #BERT max sentence len
            print('sentence too long: {} tokens'.format(labels.shape[-1]))
            return None
        labels[0][0] = labels[0][-1] = -100 #ignore on loss computation
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels, output_hidden_states=True)
        loss = outputs[0]
        correct_preds = (outputs[1][0].argmax(dim=1)[1:-1] == labels[0][1:-1]).sum().item()
        features = outputs[2][self.layer].squeeze(0)
        return loss, correct_preds, labels.shape[1]-2, features


class MLM(nn.Module):
    def __init__(self):
        super(MLM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = BertConfig(vocab_size=consts.VOCAB_SIZE)
        self.classifier = BertOnlyMLMHead(config).to(self.device)
        self.classifier.load_state_dict(torch.load('pickles/ablation/OnlyMLMHead'))
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    def count_lemma_preds(self, words_to_tokens, preds, labels, lemmas, relevant_indices):
        lemma_preds = 0
        relevant_indices = set(relevant_indices)
        for i, words_toks in enumerate(words_to_tokens):
            for j, word in enumerate(words_toks):
                if word[0] not in relevant_indices:
                    continue
                true_word = self.tokenizer.decode(labels[word[0]: word[-1] + 1])
                if not true_word in lemmas[i]:
                    continue
                pred_word = self.tokenizer.decode(preds[word[0]: word[-1] + 1])
                if pred_word.lower() != true_word.lower():
                    if lemmas[i][true_word] == pred_word:
                        lemma_preds += 1
                        print(true_word, pred_word)
        return lemma_preds

    def map_words_to_tokens(self, sentences, batch_labels):
        words_to_tokens = []
        total_tokens = 0
        for i in range(len(sentences)):
            words_to_tokens.append([])
            for token, word in enumerate(batch_labels.encodings[i].word_ids):
                if word is None:
                    continue
                if len(words_to_tokens[i]) == word:
                    words_to_tokens[i].append([token + total_tokens - 1])
                else:
                    words_to_tokens[i][word].append(token + total_tokens - 1)
            total_tokens += len(batch_labels.encodings[i]) - 2
        return words_to_tokens

    def forward(self, sentences, hidden_states, subtokens_with_attribute, lemmas):
        with torch.no_grad():
            pred_scores = self.classifier(hidden_states.to(self.device))
            batch_labels = self.tokenizer(sentences)
            labels = batch_labels['input_ids']
            labels = torch.tensor([token for tokens in labels
                                   for token in tokens[1:-1]]).to(self.device)
            loss_func = nn.CrossEntropyLoss()
            res = dict.fromkeys(['loss', 'correct_all', 'num_all', 'correct_relevant', 'num_relevant',
                                 'correct_irrelevant', 'num_irrelevant', 'lemma_preds'], None)
            res['loss'] = loss_func(pred_scores, labels).item()
            preds = pred_scores.argmax(dim=1)
            res['correct_all'] = (preds == labels).sum().item()
            res['num_all'] = labels.shape[0]
            res['correct_relevant'] = (preds[subtokens_with_attribute] == labels[subtokens_with_attribute]).sum().item()
            res['num_relevant'] = len(subtokens_with_attribute)
            irrelevant_indices = list(set(range(len(labels))) - set(subtokens_with_attribute))
            res['correct_irrelevant'] = (preds[irrelevant_indices] == labels[irrelevant_indices]).sum().item()
            res['num_irrelevant'] = len(irrelevant_indices)
            words_to_tokens = self.map_words_to_tokens(sentences, batch_labels)
            res['lemma_preds'] = \
                self.count_lemma_preds(words_to_tokens, preds, labels, lemmas, subtokens_with_attribute)
        return res

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
            else XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')
        self.layers = self.bert.bert.encoder.layer[self.layer:] if model_type == 'bert' else self.bert.roberta.encoder.layer[self.layer:]
        self.classifier = self.bert.cls if model_type == 'bert' else self.bert.lm_head
        self.prefix_char = '##' if model_type == 'bert' else '_'
        self.model_type = model_type

    def get_lemma_ranks(self, words_to_tokens, tokens_to_words, pred_scores, labels, lemmas, relevant_indices):
        res = []
        for sentence_idx, token_idxs in enumerate(relevant_indices):
            for token_idx in token_idxs:
                word_idx = tokens_to_words[sentence_idx][token_idx-1]
                word_tokens = words_to_tokens[sentence_idx][word_idx]
                # we only look at words that aren't being split, for now
                if len(word_tokens) > 1:
                    continue
                true_word = self.tokenizer.decode(labels[sentence_idx][word_tokens[0]])
                if not true_word in lemmas[sentence_idx]:
                    continue
                lemma = lemmas[sentence_idx][true_word]
                # we only look at words that are different from their lemma
                if true_word.lower() == lemma.lower():
                    continue
                lemma_token = self.tokenizer.encode(lemma)[1]
                preds = pred_scores[sentence_idx][:, word_tokens[0]].sort(descending=True).indices.tolist()
                try:
                    lemma_rank = preds.index(lemma_token)
                except:
                    lemma_rank = len(preds)
                res.append(lemma_rank)
        return res
        # for i, words_toks in enumerate(words_to_tokens):
        #     for j, word in enumerate(words_toks):
        #         # we only look at words that aren't being split, for now
        #         if len(word) > 1:
        #             continue
        #         # we only look at words that possess the attribute
        #         if word[0] not in relevant_indices[i]:
        #             continue
        #         true_word = self.tokenizer.decode(labels[i][word[0]: word[-1] + 1])
        #         if not true_word in lemmas[i]:
        #             continue
        #         lemma = lemmas[i][true_word]
        #         # we only look at words that are different from their lemma
        #         if true_word.lower() == lemma.lower():
        #             continue
        #         pred_words = self.tokenizer.decode(pred_scores[i][:,word[0]].sort(descending=True).indices).split()
        #         try:
        #             lemma_rank = pred_words.index(lemma)
        #         except:
        #             lemma_rank = len(pred_words)
        #         res.append(lemma_rank)
        #         # pred_word = self.tokenizer.decode(preds[i][word[0]: word[-1] + 1])
        #         # if pred_word.lower() != true_word.lower():
        #         #     if lemmas[i][true_word] == pred_word:
        #         #         lemma_preds += 1
        #         #         print(true_word, pred_word)
        # return res

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
                    if self.model_type == 'bert' and not curr_token.startswith(self.prefix_char):
                        splits += 1
                    elif self.model_type == 'xlm' and curr_token.startswith(self.prefix_char):
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
            word_idx = -1
            output_word_idx = -1
            for token_idx, token in enumerate(pred_sentence_tokens):
                if token == "We":
                    print('here')
                if not token.startswith('##'):
                    output_word_idx += 1
                curr_token_input_word = input_tokens_to_words[sent_idx][token_idx]
                if len(input_to_output[sent_idx]) == curr_token_input_word:
                    input_to_output[sent_idx].append({output_word_idx})
                else:
                    input_to_output[sent_idx][curr_token_input_word].add(output_word_idx)
        return input_to_output


    def forward(self, sentences, features, subtokens_with_attribute=None, lemmas=None):
        with torch.no_grad():
            batch_labels = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.device)
            words_to_tokens = self.map_words_to_tokens(sentences, batch_labels)
            tokens_to_words = self.map_tokens_to_words(words_to_tokens)
            labels = batch_labels['input_ids']
            attention_mask = torch.where(labels == 0, torch.zeros_like(labels), torch.ones_like(labels))
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            padded_features = torch.zeros(list(labels.shape) + [consts.BERT_OUTPUT_DIM]).to(self.device)
            for i in range(len(features)):
                padded_features[i, :features[i].shape[0], :] = features[i]
            # features = features.to(self.device).unsqueeze(0)
            for layer in self.layers:
                padded_features = layer(padded_features, attention_mask=extended_attention_mask)[0]
            pred_scores = self.classifier(padded_features)
            transpose = []
            for sentence_scores in pred_scores:
                transpose.append(sentence_scores.T)
            pred_scores = torch.stack(transpose)
            preds = pred_scores.argmax(dim=1)
            # pred_scores = pred_scores[0].T.unsqueeze(0)
            labels = torch.where(labels == 101, torch.ones_like(labels) * (-100), labels)
            labels = torch.where(labels == 102, torch.ones_like(labels) * (-100), labels)
            labels = torch.where(labels == 0, torch.ones_like(labels) * (-100), labels)
            loss_func = nn.CrossEntropyLoss()
            res = dict.fromkeys(['loss', 'correct_all', 'num_all', 'correct_relevant', 'num_relevant',
                                 'correct_irrelevant', 'num_irrelevant','lemma_preds'],None)
            res['lemmas_ranks'] = \
                self.get_lemma_ranks(words_to_tokens, tokens_to_words, pred_scores,labels,lemmas, subtokens_with_attribute)
            res['loss'] = loss_func(pred_scores, labels).item()
            res['correct_all'] = (preds == labels).sum().item()
            res['num_all'] = (labels != -100).sum().item()
            res['correct_relevant'], res['num_relevant'] = \
                specific_words_acc(subtokens_with_attribute, preds, labels, True)
            res['correct_irrelevant'], res['num_irrelevant'] = \
                specific_words_acc(subtokens_with_attribute, preds, labels, False)
            # res['pred_sentences'] = [self.tokenizer.decode(preds[i][labels[i]>0]) for i in range(len(sentences))]
            # self.map_input_to_output(labels, preds, tokens_to_words)
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

class PosTaggerSingleNeuron(PosTagger):
    def __init__(self, first_layer_size=0):
        super(PosTaggerSingleNeuron, self).__init__(first_layer_size)
        self.fc1 = nn.Linear(1, consts.LABEL_DIM).to(self.device)
    def forward(self, feature):
        preds = self.fc1(feature)
        return preds

class PosTaggerSingleWithHidden(PosTagger):
    def __init__(self, first_layer_size=0):
        super(PosTaggerSingleWithHidden, self).__init__(first_layer_size)
        self.fc1 = nn.Linear(1, consts.HIDDEN_LAYER_DIM).to(self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(consts.HIDDEN_LAYER_DIM, consts.BERT_OUTPUT_DIM).to(self.device)
    def forward(self, feature):
        preds = self.fc2(self.relu(self.fc1(feature)))
        return preds

class SinglePosPredictor(PosTagger):
    def __init__(self,first_layer_size=0):
        super(SinglePosPredictor, self).__init__(first_layer_size)
        self.fc1 = nn.Linear(1, 2).to(self.device)
    def forward(self, feature):
        preds = self.fc1(feature)
        return preds

class PosTaggerSubset(PosTagger):
    def __init__(self, first_layer_size, num_labels):
        super(PosTaggerSubset, self).__init__()
        self.fc1 = nn.Linear(first_layer_size, num_labels)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(HIDDEN_LAYER_DIM, LABEL_DIM)
    def forward(self, features):
        # preds=self.fc2(self.relu(self.fc1(features)))
        preds = self.fc1(features)
        return preds
