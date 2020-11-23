
python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
  --output_path data/swow/concept_bert_emb.npy\
  --max_seq_length 32\
  --device 6\
  --batch_size 256\

#   --debug