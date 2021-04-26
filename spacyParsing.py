
import pickle
import sys
from argparse import ArgumentParser

import spacy
from pathlib import Path
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from spacy.tokens import Doc
from transformers import BertTokenizer

class morphCompare():
    def __init__(self, language, attribute, layer, ranking):
        self.language = language
        self.attribute = attribute
        self.layer = layer
        self.ranking = ranking
        parsers = {'eng':'en_core_web_sm',
                   'rus':'ru_core_news_sm'}
        self.parser = spacy.load(parsers[self.language])
        # self.parser.tokenizer = BertTokenizer(self.parser.vocab, "bert-base-multilingual-cased-vocab.txt")
        self.true_morph, self.true_words_to_tokens, self.true_tokens_to_words = self.parse_true()
        self.pred_morph, self.pred_words_to_tokens, self.pred_tokens_to_words = self.parse_preds()

    def _words_to_tokens(self, sentences: list):
        sentences_by_tokens = []
        tokens_to_words = []
        for sentence in sentences:
            sentence_by_tokens = {}
            toks_to_words = {}
            token_idx = 0
            for word_idx, word in enumerate(sentence.split()):
                word_tokens_len = len(self.parser.tokenizer(word))
                word_tokens_idxs= list(range(token_idx, token_idx + word_tokens_len))
                sentence_by_tokens[word_idx]=word_tokens_idxs
                for token in word_tokens_idxs:
                    toks_to_words[token] = word_idx
                token_idx += word_tokens_len
            sentences_by_tokens.append(sentence_by_tokens)
            tokens_to_words.append(toks_to_words)
        return sentences_by_tokens, tokens_to_words


    def parse(self, sentences):
        stats = {}
        for sent_idx, sentence in enumerate(sentences):
            parsed_sentence = self.parser(sentence)
            sentence_stats = {'ids': {},'lemmas': {}, 'attribute': {}}
            for token_idx, token in enumerate(parsed_sentence):
                sentence_stats['ids'][token_idx] = token.text
                sentence_stats['lemmas'][token_idx] = token.lemma_
                if self.attribute == "Part of Speech":
                    sentence_stats['attribute'][token_idx] = token.pos_
                morph = token.morph.to_dict()
                if self.attribute == "Gender and Noun Class":
                    #mismatch between UM and Spacy notations
                    if 'Gender' in morph:
                        sentence_stats['attribute'][token_idx] = morph['Gender']
                elif self.attribute in morph:
                    sentence_stats['attribute'][token_idx] = morph[self.attribute]
            stats[sent_idx] = sentence_stats
        return stats

    def parse_true(self):
        true_sentences_path = Path('pickles', 'UM', self.language, 'new_test_sentences.pkl')
        # true_sentences_path = Path('pickles', 'UM', self.language, 'test_sentences.pkl')
        with open(true_sentences_path, 'rb') as f:
            true_sentences = pickle.load(f)
        true_sentences = list(true_sentences.items())
        true_sentences.sort(key= lambda x: x[0])
        true_sentences = [t[1] for t in true_sentences]
        # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        # true_sentences = [t[1][:-1] for t in true_sentences] # [-1] is for removing \n
        parsed = self.parse(true_sentences)
        tokenization, rev_tokenization = self._words_to_tokens(true_sentences)
        #TODO remove after debug
        # self.true_sentences = true_sentences
        return parsed, tokenization, rev_tokenization

    def parse_preds(self):
        pred_sentences_path = Path('pickles', 'UM', self.language, self.attribute, str(self.layer), 'ablation_token_outputs_by_' + self.ranking + '.pkl')
        with open(pred_sentences_path, 'rb') as f:
            pred_sentences = pickle.load(f)
        pred_stats, pred_tokenization, pred_rev_tokenization = {}, {}, {}
        # #TODO remove after debug
        # pred_sentences = pred_sentences[400]
        # pred_sentences = {400: pred_sentences}
        for num_ablated, preds in tqdm(pred_sentences.items()):
            joined_preds = [' '.join(words) for words in preds]
            pred_stats[num_ablated] = self.parse(joined_preds)
            pred_tokenization[num_ablated], pred_rev_tokenization[num_ablated] = self._words_to_tokens(joined_preds)
        return pred_stats, pred_tokenization, pred_rev_tokenization

    def comp_stats(self, num_ablated):
        stats = dict.fromkeys(['correct word', 'wrong word', 'correct lemma', 'wrong lemma',
                               'kept attribute', 'no attribute', 'correct val', 'wrong val', 'relevant',
                               'true split', 'pred split',
                               'correct lemma, correct value', 'correct lemma, wrong value',
                               'wrong lemma, correct value', 'wrong lemma, wrong value'], 0)
        pred_morph = self.pred_morph[num_ablated]
        pred_tokenization = self.pred_words_to_tokens[num_ablated]
        for sent_idx, curr_stats in self.true_morph.items():
            prev_word_idx = -1
            for token_idx, val in curr_stats['attribute'].items():
                stats['relevant'] += 1
                word_idx = self.true_tokens_to_words[sent_idx][token_idx]
                true_word_tokens = self.true_words_to_tokens[sent_idx][word_idx]
                pred_token_idx = pred_tokenization[sent_idx][word_idx]
                if len(true_word_tokens) != len(pred_token_idx):
                    stats['pred split'] += 1
                    if len(true_word_tokens) > 1:
                        stats['true split'] += 1
                    if prev_word_idx != word_idx:
                        true_words = [curr_stats['ids'][tok] for tok in true_word_tokens]
                        pred_words = [pred_morph[sent_idx]['ids'][tok] for tok in pred_token_idx]
                        print(f"true words: {' '.join(true_words)}")
                        print(f"pred words: {' '.join(pred_words)}")
                        prev_word_idx = word_idx
                    continue
                pred_token_idx = pred_token_idx[true_word_tokens.index(token_idx)]
                if curr_stats['ids'][token_idx] == pred_morph[sent_idx]['ids'][pred_token_idx]:
                    stats['correct word'] += 1
                else:
                    stats['wrong word'] += 1
                    if curr_stats['lemmas'][token_idx].lower() == pred_morph[sent_idx]['lemmas'][pred_token_idx].lower():
                        stats['correct lemma'] += 1
                        if pred_token_idx in pred_morph[sent_idx]['attribute'].keys() and val == pred_morph[sent_idx]['attribute'][pred_token_idx]:
                            stats['correct lemma, correct value'] += 1
                        else:
                            stats['correct lemma, wrong value'] += 1
                    else:
                        stats['wrong lemma'] += 1
                        if pred_token_idx in pred_morph[sent_idx]['attribute'].keys() \
                            and val == pred_morph[sent_idx]['attribute'][pred_token_idx]:
                            stats['wrong lemma, correct value'] += 1
                        else:
                            stats['wrong lemma, wrong value'] += 1
                    if pred_token_idx in pred_morph[sent_idx]['attribute'].keys():
                        stats['kept attribute'] += 1
                        if val == pred_morph[sent_idx]['attribute'][pred_token_idx]:
                            stats['correct val'] += 1
                        else:
                            stats['wrong val'] += 1
                    else:
                        stats['no attribute'] += 1
                prev_word_idx = word_idx
        print(stats)

    def comp_all(self):
        for num_ablated in self.pred_morph.keys():
            print(f'num ablated: {num_ablated}')
            self.comp_stats(num_ablated)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-language', type=str)
    argparser.add_argument('-attribute', type=str)
    argparser.add_argument('-layer', type=int)
    argparser.add_argument('-ranking', type=str)
    args = argparser.parse_args()
    language = args.language
    attribute = args.attribute
    layer = args.layer
    ranking = args.ranking
    res_dir = Path('results','UM', language, attribute, 'layer '+str(layer), 'spacy')
    if not res_dir.exists():
        res_dir.mkdir()
    res_file_name = 'by ' + ranking
    with open(Path(res_dir, res_file_name), 'w+') as f: ###############TODO
        sys.stdout = f
        print('language: ', language)
        print('attribute: ', attribute)
        print('layer: ', layer)
        print('ranking: ', ranking)
        mc = morphCompare(language, attribute, layer, ranking)
        mc.comp_all()
