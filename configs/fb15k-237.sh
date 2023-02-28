#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/FB15K-237/"
vocab_dir="datasets/data_preprocessed/FB15K-237/vocab"
emb_dir="datasets/data_preprocessed/FB15K-237/embeddings/"
total_iterations=2000
path_length=3
hidden_size=100
embedding_size=100
batch_size=256
beta=0.02
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/fb15k-237/"
load_model=0
model_load_dir="saved_models/fb15k-237"
nell_evaluation=0
