import os
import logging as logger
import argparse
import csv
import json
import random
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

try:
    from .utils import check_path
except:
    from utils import check_path


def extract_bert_embeddings(dataset, node_file, output_dir):

    node_list = load_nodes(node_file)
    #node_list = graph.iter_nodes()
    bert_model = BertLayer(dataset)
    embeddings = bert_model.forward(node_list)

    output_file  = os.path.join(output_dir, dataset + "_bert_embeddings.pt")
    torch.save(embeddings, output_file)
    print("Writing {}".format(output_file))

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]

def convert_concept_to_bert_input(tokenizer, concept, max_seq_length=32):
    concept_tokens = tokenizer.tokenize(concept)
    concept_tokens  = concept_tokens[:max_seq_length - len(concept_tokens) - 2]
    tokens = [tokenizer.cls_token] +  concept_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(tokens))

    assert len(input_ids) == len(segment_ids) == len(input_ids)

    # padding
    pad_len = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids += [0] * pad_len
    segment_ids += [0] * pad_len
    span = (1, 1 + len(concept_tokens))

    assert span[1] + 1 == len(tokens)
    assert max_seq_length == len(input_ids) == len(segment_ids) == len(input_mask)

    return input_ids, input_mask, segment_ids, span

def extract_bert_node_features_for_concepts(cpnet_vocab_path, output_path, max_seq_length, device, batch_size, layer_id=-1, cache_path=None, use_cache=True, debug=False):
    global id2concept
    # if id2concept is None:
    load_resources(cpnet_vocab_path=cpnet_vocab_path)
    check_path(output_path)

    print(f'extracting concepts embeddings from {cpnet_vocab_path}')

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(device)
    model.eval()
    
    all_input_ids, all_input_mask, all_segment_ids, all_span = [], [], [], []
    if debug:
        id2concept=id2concept[:10]
    n = len(id2concept)
    for concept in tqdm(id2concept, total=n, desc='Calculating input features'):
        concept = concept.replace('_', ' ') 
        input_ids, input_mask, segment_ids, span = convert_concept_to_bert_input(tokenizer, concept, max_seq_length)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_span.append(span)

    all_input_ids, all_input_mask, all_segment_ids, all_span = [torch.tensor(x, dtype=torch.long) for x in [all_input_ids, all_input_mask, all_segment_ids, all_span]]
    all_span = all_span.to(device)

    concept_vecs = []
    n = all_input_ids.size(0)

    with torch.no_grad():
        for a in tqdm(range(0, n, batch_size), total=n // batch_size + 1, desc='Extracting features'):
            b = min(a + batch_size, n)
            batch = [x.to(device) for x in [all_input_ids[a:b], all_input_mask[a:b], all_segment_ids[a:b]]]
            outputs = model(*batch)
            hidden_states = outputs[-1][layer_id]
            mask = torch.arange(max_seq_length, device=device)[None, :]
            mask = (mask >= all_span[a:b, 0, None]) & (mask < all_span[a:b, 1, None])
            pooled = (hidden_states * mask.float().unsqueeze(-1)).sum(1)
            pooled = pooled / (all_span[a:b, 1].float() - all_span[a:b, 0].float() + 1e-5).unsqueeze(1)
            concept_vecs.append(pooled.cpu())
        concept_vecs = torch.cat(concept_vecs, 0).numpy()

    check_path(output_path)
    res = np.array(concept_vecs , dtype="float32")
    np.save(output_path, res)
    print(f'save {output_path}, {res.shape()}')
    print('done!')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')
    parser.add_argument("--cpnet_vocab_path", type=str, required=True, help="nodes for trianing bert model")
    parser.add_argument("--output_path", type=str, required=True, help="output directory to store metrics and model file")
    parser.add_argument("--device", type=int, required=True, help="gpu device number")
    parser.add_argument("--max_seq_length", type=int, default=32, help="max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--debug", action='store_true', help="batch size")

    args = parser.parse_args()

    extract_bert_node_features_for_concepts(args.cpnet_vocab_path, args.output_path, args.max_seq_length, args.device, args.batch_size, layer_id=-1, cache_path=None, use_cache=True, debug=args.debug)


