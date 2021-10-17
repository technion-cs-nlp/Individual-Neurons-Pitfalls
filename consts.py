import numpy as np
import pickle
INF = np.inf
EPOCHS=10
NUM_SENTENCES_TRAIN=5000
NUM_SENTENCES_TEST=1000
BATCH_SIZE=32
ABLATION_BATCH_SIZE=8
LEARNING_RATE=0.01
BERT_OUTPUT_DIM = 768
LABEL_DIM = 17
HIDDEN_LAYER_DIM=128
SEED=1
EPSILON=0.01
SUBSET_SIZE = 150
SMALL_DATA_SIZE = {'train_': 5000, 'dev_': 1000, 'test_': 1000}
# VOCAB_SIZE = 28996
VOCAB_SIZE = 119547
ABLATION_NUM_SENTENCES = 6500
ABLATION_NUM_TOKENS = 100000
UM_FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

# penn_to_ud_labels = {'#': 'SYM',
#                      '$': 'SYM',
#                      '"': 'PUNCT',
#                      ',': 'PUNCT',
#                      '-LRB-': 'PUNCT',
#                      '-RRB-': 'PUNCT',
#                      '.': 'PUNCT',
#                      ':': 'PUNCT',
#                      'AFX': 'ADJ',
#                      'CC': 'CCONJ',
#                      'CD': 'NUM',
#                      'DT': 'DET',
#                      'EX': 'PRON',
#                      'FW': 'X',
#                      'HYPH': 'PUNCT',
#                      'IN': 'ADP',
#                      'JJ': 'ADJ',
#                      'JJR': 'ADJ',
#                      'JJS': 'ADJ',
#                      'LS': 'X',
#                      'MD': 'VERB',
#                      'NIL': 'X',
#                      'NN': 'NOUN',
#                      'NNP': 'PROPN',
#                      'NNPS': 'PROPN',
#                      'NNS': 'NOUN',
#                      'PDT': 'DET',
#                      'POS': 'PART',
#                      'PRP': 'PRON',
#                      'PRP$': 'DET',
#                      'RB': 'ADV',
#                      'RBR': 'ADV',
#                      'RBS': 'ADV',
#                     'RP': 'ADP',
#                     'SYM': 'SYM',
#                     'TO': 'PART',
#                     'UH': 'INTJ',
#                     'VB': 'VERB',
#                     'VBD': 'VERB',
#                     'VBG': 'VERB',
#                     'VBN': 'VERB',
#                     'VBP': 'VERB',
#                     'VBZ': 'VERB',
#                     'WDT': 'DET',
#                     'WP': 'PRON',
#                     'WP$': 'DET',
#                     'WRB': 'ADV',
#                     "''": 'PUNCT',
#                      '``': 'PUNCT'
#                      }
train_paths = {'eng': 'data/UM/eng/en_ewt-um-train.conllu',
               'ara': 'data/UM/ara/ar_padt-um-train.conllu',
               'hin': 'data/UM/hin/hi_hdtb-um-train.conllu',
               'rus': 'data/UM/rus/ru_gsd-um-train.conllu',
               'fin': 'data/UM/fin/fi_tdt-um-train.conllu',
               'bul': 'data/UM/bul/bg_btb-um-train.conllu',
               'tur': 'data/UM/tur/tr_imst-um-train.conllu',
               'spa': 'data/UM/spa/es_gsd-um-train.conllu',
               'fra': 'data/UM/fra/fr_gsd-um-train.conllu'}

dev_paths = {'eng': 'data/UM/eng/en_ewt-um-dev.conllu',
             'ara': 'data/UM/ara/ar_padt-um-dev.conllu',
             'hin': 'data/UM/hin/hi_hdtb-um-dev.conllu',
             'rus': 'data/UM/rus/ru_gsd-um-dev.conllu',
             'fin': 'data/UM/fin/fi_tdt-um-dev.conllu',
             'bul': 'data/UM/bul/bg_btb-um-dev.conllu',
             'tur': 'data/UM/tur/tr_imst-um-dev.conllu',
             'spa': 'data/UM/spa/es_gsd-um-dev.conllu',
             'fra': 'data/UM/fra/fr_gsd-um-dev.conllu'}

test_paths = {'eng': 'data/UM/eng/en_ewt-um-test.conllu',
              'ara': 'data/UM/ara/ar_padt-um-test.conllu',
              'hin': 'data/UM/hin/hi_hdtb-um-test.conllu',
              'rus': 'data/UM/rus/ru_gsd-um-test.conllu',
              'fin': 'data/UM/fin/fi_tdt-um-test.conllu',
              'bul': 'data/UM/bul/bg_btb-um-test.conllu',
              'tur': 'data/UM/tur/tr_imst-um-test.conllu',
              'spa': 'data/UM/spa/es_gsd-um-test.conllu',
              'fra': 'data/UM/fra/fr_gsd-um-test.conllu'}