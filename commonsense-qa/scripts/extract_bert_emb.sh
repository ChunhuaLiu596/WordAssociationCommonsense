#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
  # --output_path data/swow/\
  # --model_name 'bert-large-uncased'\
  # --max_seq_length 32\
  # --device 0\
  # --batch_size 1\
  # --debug

# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
  # --output_path data/swow/\
  # --model_name 'bert-large-uncased'\
  # --max_seq_length 32\
  # --device 0\
  # --batch_size 1  
# 

### cpnet
# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet/concept.txt\
  # --output_path data/cpnet/\
  # --model_name 'roberta-large'\
  # --max_seq_length 32\
  # --device 0\
  # --batch_size 32

# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet/concept.txt\
#   --output_path data/cpnet/\
#   --model_name 'bert-large-uncased'\
#   --max_seq_length 32\
#   --device 0\
#   --batch_size 32



# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet/concept.txt\
#   --output_path data/cpnet/\
#   --model_name 'albert-xxlarge-v2'\
#   --max_seq_length 32\
#   --device 0\
#   --batch_size 4

### swow
# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
#   --output_path data/swow/\
#   --model_name 'roberta-large'\
#   --max_seq_length 32\
#   --device 0\
#   --batch_size 32

# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
#   --output_path data/swow/\
#   --model_name 'bert-large-uncased'\
#   --max_seq_length 32\
#   --device 0\
#   --batch_size 32



# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
#   --output_path data/swow/\
#   --model_name 'albert-xxlarge-v2'\
#   --max_seq_length 32\
#   --device 0\
#   --batch_size 4


# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
  # --output_path data/swow/\
  # --model_name 'bert-large-uncased'\
  # --max_seq_length 32\
  # --device 0\
  # --batch_size 1\
  # --debug

# python utils/extract_bert_embeddings.py --cpnet_vocab_path data/swow/concept.txt\
  # --output_path data/swow/\
  # --model_name 'bert-large-uncased'\
  # --max_seq_length 32\
  # --device 0\
  # --batch_size 1  
# 

### cpnet
python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet7rel/concept.txt\
  --output_path data/cpnet7rel/\
  --model_name 'roberta-large'\
  --max_seq_length 32\
  --device 0\
  --batch_size 32
# 
python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet7rel/concept.txt\
  --output_path data/cpnet7rel/\
  --model_name 'bert-large-uncased'\
  --max_seq_length 32\
  --device 0\
  --batch_size 32
# 
python utils/extract_bert_embeddings.py --cpnet_vocab_path data/cpnet7rel/concept.txt\
  --output_path data/cpnet7rel/\
  --model_name 'albert-xxlarge-v2'\
  --max_seq_length 32\
  --device 0\
  --batch_size 4
# 