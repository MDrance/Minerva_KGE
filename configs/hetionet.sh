#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/hetionet/"
vocab_dir="datasets/data_preprocessed/hetionet/vocab"
emb_dir="datasets/data_preprocessed/hetionet/embeddings/"
total_iterations=1000
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/hetionet/"
load_model=0
model_load_dir="null"
nell_evaluation=0
