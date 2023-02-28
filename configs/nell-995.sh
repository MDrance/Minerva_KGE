#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell-995/"
vocab_dir="datasets/data_preprocessed/nell-995/vocab"
emb_dir="datasets/data_preprocessed/nell-995/embeddings/"
total_iterations=3000
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/nell-995/worksfor"
load_model=0
model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/nnnn/45de_3_0.06_10_0.0/model/model.ckpt"
nell_evaluation=0
