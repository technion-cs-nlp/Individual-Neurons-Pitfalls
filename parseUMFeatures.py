from argparse import ArgumentParser

from conllu import parse_incr, TokenList
import consts
import yaml
from typing import List, Tuple, Dict, Any
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle
from os import path
import torch
from torch_scatter import scatter_mean
from pathlib import Path


"""
This function preprocesses UniMorph annotations as described in App. C on the paper.
"""
def parse_unimorph_features(features: List[str]) -> Dict[str, str]:
    final_attrs: Dict[str, str] = {}
    for x in features:
        if "/" in x:
            # Drop disjunctions.
            continue
        elif x == "{CMPR}":
            # I am assuming they meant to type "CMPR" and not "{CMPR}"
            final_attrs["Comparison"] = "CMPR"
        elif x == "PST+PRF":
            # PST+PRF is a common feature of Latin, Romanian, and Turkish annotations.
            # I assign it to Tense due aspect having already been assigned to something different in Turkish,
            # and since "PST" comes first.
            final_attrs["Tense"] = x
        elif x.startswith("ARG"):
            # Argument marking (e.g. in Basque) is labelled with ARGX where X is the actual feature.
            v = x[3:]
            final_attrs[_UNIMORPH_VALUES_ATTRIBUTE[v]] = v
        elif x == "NDEF":
            final_attrs["Definiteness"] = "INDF"
        elif "+" in x:
            # We handle conjunctive statements by creating a new value for them.
            # We canonicalize the feature by sorting the composing conjuncts alphabetically.
            values = x.split("+")
            attr = _UNIMORPH_VALUES_ATTRIBUTE[values[0]]
            for v in values:
                if attr != _UNIMORPH_VALUES_ATTRIBUTE[v]:
                    raise Exception("Conjunctive values don't all belong to the same dimension.")

            final_attrs[attr] = "+".join(sorted(values))
        elif "PSS" in x:
            final_attrs["Possession"] = x
        elif "LGSPEC" in x:
            # We discard language-specific features as this is not a canonical unimorph dimension
            continue
        elif x == "V" and "V." in [f[:2] for f in features]:
            continue
        else:
            if x not in _UNIMORPH_VALUES_ATTRIBUTE:
                continue
            final_attrs[_UNIMORPH_VALUES_ATTRIBUTE[x]] = x
    return final_attrs

def unimorph_feature_parser(line: List[str], i: int) -> Dict[str, str]:
    if line[i] == "_":
        return {}

    return parse_unimorph_features(line[i].split(";"))

def merge_attributes(tokens: List[str], value_to_attr_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Returns a dictionary containing Unimorph attributes, and the values taken on after the merge.
    """
    # First, build a list that naively merges everything
    merged_attributes: Dict[str, List[str]] = {}
    for t in tokens:
        for attr, val in t["um_feats"].items():
            if attr not in merged_attributes:
                merged_attributes[attr] = []

            merged_attributes[attr].append(val)

    # Second, remove attributes with multiple values (even if they are the same)
    final_attributes: Dict[str, str] = {}
    for attr, vals in merged_attributes.items():
        if len(vals) == 1:
            final_attributes[attr] = vals[0]

    return final_attributes

def subword_tokenize(tokenizer: BertTokenizer, tokens: List[str]) -> List[Tuple[int, str]]:
    """
    Returns: List of subword tokens, List of indices mapping each subword token to one real token.
    """
    subtokens = [tokenizer.tokenize(t) for t in tokens]

    indexed_subtokens = []
    for idx, subtoks in enumerate(subtokens):
        for subtok in subtoks:
            indexed_subtokens.append((idx, subtok))

    return indexed_subtokens

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

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-language', type=str)
    args = argparser.parse_args()
    language = args.language
    print(f'language: {language}')
    train_path = train_paths[language]
    dev_path = dev_paths[language]
    test_path = test_paths[language]
    pickles_root = Path('pickles','UM',language)
    if not pickles_root.exists():
        pickles_root.mkdir()
    train_dump_path = Path(pickles_root,'train_parsed.pkl')
    dev_dump_path = Path(pickles_root,'dev_parsed.pkl')
    test_dump_path = Path(pickles_root,'test_parsed.pkl')
    train_sentences_path = Path(pickles_root,'new_train_sentences.pkl')
    dev_sentences_path = Path(pickles_root, 'new_dev_sentences.pkl')
    test_sentences_path = Path(pickles_root, 'new_test_sentences.pkl')
    train_attributes_path = Path(pickles_root, 'train_words_per_attribute.pkl')
    dev_attributes_path = Path(pickles_root, 'dev_words_per_attribute.pkl')
    test_attributes_path = Path(pickles_root, 'test_words_per_attribute.pkl')
    train_lemmas_path = Path(pickles_root, 'train_lemmas.pkl')
    dev_lemmas_path = Path(pickles_root, 'dev_lemmas.pkl')
    test_lemmas_path = Path(pickles_root, 'test_lemmas.pkl')
    tags_file = "unimorph/tags.yaml"
    skip_existing = False
    bert_name = 'bert-base-multilingual-cased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0

    with open(tags_file, 'r') as h:
        _UNIMORPH_ATTRIBUTE_VALUES = yaml.full_load(h)["categories"]

    _UNIMORPH_VALUES_ATTRIBUTE = {v: k for k, vs in _UNIMORPH_ATTRIBUTE_VALUES.items() for v in vs}
    limit_number = None
    skipped: Dict[str, int] = {}
    # Setup BERT tokenizer here provisionally as we need to know which sentences have over 512 subtokens
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    for file_path, dump_path, sentences_path, attributes_path, lemmas_path in zip(
            [test_path, dev_path, train_path],
            [test_dump_path, dev_dump_path, train_dump_path],
            [test_sentences_path, dev_sentences_path, train_sentences_path],
            [test_attributes_path, dev_attributes_path, train_attributes_path],
            [test_lemmas_path, dev_lemmas_path, train_lemmas_path]):
        final_token_list: List[TokenList] = []
        final_results = []
        with open(file_path,'r') as h:
            for sent_id, tokenlist in enumerate(tqdm(
                    parse_incr(h, fields=consts.UM_FEATS, field_parsers={"um_feats": unimorph_feature_parser}))):
                # Only process first `limit_number` if it is set
                if limit_number is not None and sent_id > limit_number:
                    break

                # Remove virtual nodes
                tokenlist = [t for t in tokenlist if not (isinstance(t["id"], tuple) and t["id"][1] == ".")]

                # Build list of ids that are contracted
                contracted_ids: List[int] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        if t["id"][1] == "-":
                            # Range
                            contracted_ids.extend(list(range(t["id"][0], t["id"][2] + 1)))

                # Build dictionary of non-contracted token ids to tokens
                non_contracted_token_dict: Dict[int, Any] = {
                    t["id"]: t for t in tokenlist if not isinstance(t["id"], tuple)}

                # Build final list of (real) tokens, without any contractions
                # Contractions are assigned the attributes of the constituent words, unless there is a clash
                # with one attribute taking more than one value (e.g. POS tag is a frequent example), whereby
                # we discard it.
                final_tokens: List[Any] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        constituent_ids = list(range(t["id"][0], t["id"][2] + 1))
                        t["um_feats"] = merge_attributes(
                            [non_contracted_token_dict[x] for x in constituent_ids],
                            _UNIMORPH_VALUES_ATTRIBUTE)

                        # If this is a contraction, add it
                        final_tokens.append(t)
                    elif t["id"] not in contracted_ids:
                        # Check if this t is part of a contraction
                        final_tokens.append(t)

                final_tokens: TokenList = TokenList(final_tokens)

                # Skip if this would have more than 512 subtokens
                labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in final_tokens])
                subtoken_indices, subtokens = zip(*labelled_subwords)
                if len(subtokens) >= 512:
                    if "subtoken_count" not in skipped:
                        skipped["subtoken_count"] = 0

                    skipped["subtoken_count"] += 1
                    continue

                if "total_sents" not in skipped:
                    skipped["total_sents"] = 0

                # Add this sentence to the list we are processing
                final_token_list.append(final_tokens)
        # with open(dump_path,'wb+') as f:
        #     pickle.dump(final_token_list, f, pickle.HIGHEST_PROTOCOL)
        print("Skipped:")
        print(skipped)
        print()

        if skip_existing and path.exists(dump_path):
            print(f"Skipping {file_path}. Reason: file already processed")
            continue

        print(f"Processing {file_path}...")
        # Setup BERT
        model = BertModel.from_pretrained(bert_name).to(device)
        # Subtokenize, keeping original token indices
        results = []
        sentences = {}
        num_sentences = 0
        lemmas = []
        for sent_id, tokenlist in enumerate(tqdm(final_token_list)):
            sentences[num_sentences] = ' '.join([word['form'] for word in tokenlist])
            num_sentences += 1
            labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in tokenlist])
            subtoken_indices, subtokens = zip(*labelled_subwords)
            subtoken_indices_tensor = torch.tensor(subtoken_indices).to(device)

            # We add special tokens to the sequence and remove them after getting the BERT output
            subtoken_ids = torch.tensor(
                tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(subtokens))).to(device)

            results.append((tokenlist, subtoken_ids, subtoken_indices_tensor))

            # Add sentence lemmas to lemmas list
            lemmas.append({token['form']: token['lemma'] for token in tokenlist})

        with open(sentences_path, 'wb+') as f:
            pickle.dump(sentences, f, pickle.HIGHEST_PROTOCOL)
        with open(lemmas_path,'wb+') as f:
            pickle.dump(lemmas, f)
        # Prepare to compute BERT embeddings
        model.eval()

        # NOTE: No batching, right now. But could be worthwhile to implement if a speed-up is necessary.
        # subtokens_per_sentence = {}
        subtokens_for_attribute_for_sentence = {}
        for i, (token_list, subtoken_ids, subtoken_indices_tensor) in tqdm(enumerate(results)):
            total += 1
            real_subtokens = (subtoken_indices_tensor + 1).tolist()
            subtokens_to_words = {j:[] for j in range(len(token_list)+2)}
            [subtokens_to_words[token].append(j+1) for j, token in enumerate(real_subtokens)]
            # subtokens_per_sentence[i] = subtokens_to_words
            subtokens_for_attribute_for_sentence[i] = {}
            for token_idx, t in enumerate(token_list):
                for attribute, label in t['um_feats'].items():
                    if attribute not in subtokens_for_attribute_for_sentence[i].keys():
                        subtokens_for_attribute_for_sentence[i][attribute] = {}
                    if label not in subtokens_for_attribute_for_sentence[i][attribute]:
                        subtokens_for_attribute_for_sentence[i][attribute][label] = []
                    subtokens_for_attribute_for_sentence[i][attribute][label].extend(subtokens_to_words[token_idx+1])
            with torch.no_grad():
                # shape: (batch_size, max_seq_length_in_batch + 2)
                inputs = subtoken_ids.reshape(1, -1)

                # shape: (batch_size, max_seq_length_in_batch)
                indices = subtoken_indices_tensor.reshape(1, -1)

                # shape: (batch_size, max_seq_length_in_batch + 2, embedding_size)
                outputs = model(inputs, output_hidden_states=True)
                # final_output = outputs[0]
                final_output = torch.stack([hidden_state[0][1:-1] for hidden_state in outputs.hidden_states])
                # final_output = final_output.permute(1,0,-1)
                # shape: (batch_size, max_seq_length_in_batch, embedding_size)
                # Here we remove the special tokens (BOS, EOS)
                # final_output = torch.stack([output[:, 1:, :][:, :-1, :] for output in final_output])

                # Average subtokens corresponding to the same word
                # shape: (batch_size, max_num_tokens_in_batch, embedding_size)
                token_embeddings = scatter_mean(final_output, indices, dim=1)
                token_embeddings = token_embeddings.permute(1,0,-1)

            # Convert to python objects
            embedding_list = [x.squeeze(0).cpu().numpy() for x in token_embeddings.split(1, dim=0)]

            for t, e in zip(token_list, embedding_list):
                t["embedding"] = e

            final_results.append(token_list)

        final_results_filtered = []
        for row in final_results:
            for token in row:
                final_results_filtered.append({
                    "word": token["form"],
                    "embedding": token["embedding"],
                    "attributes": token["um_feats"],
                })

        # Save final results
        with open(dump_path, "wb+") as h:
            pickle.dump(final_results_filtered, h)

        with open(attributes_path, 'wb+') as h:
            pickle.dump(subtokens_for_attribute_for_sentence,h)