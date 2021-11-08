import numpy as np
import pickle
INF = np.inf
EPOCHS=10
BATCH_SIZE=32
ABLATION_BATCH_SIZE=8
LEARNING_RATE=0.01
BERT_OUTPUT_DIM = 768
SEED=1
SUBSET_SIZE = 150
# VOCAB_SIZE = 28996
VOCAB_SIZE = 119547
ABLATION_NUM_SENTENCES = 6500
ABLATION_NUM_TOKENS = 100000
UM_FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

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