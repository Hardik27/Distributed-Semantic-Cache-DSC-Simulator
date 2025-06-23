# Distributed-Semantic-Cache-DSC-Simulator

This repository contains a discrete-event simulation of a distributed semantic caching mechanism for Retrieval-Augmented Generation (RAG) systems.

## 📌 Overview

RAG systems often face redundant computation due to repeated or semantically similar queries across distributed nodes. This simulator models:

- Local semantic caching with cosine similarity
- Bloom filter–based coordination across nodes
- Cache performance comparison between:
  - Centralized Exact Cache (CEC)
  - Isolated Semantic Cache (IC)
  - Distributed Semantic Cache (DSC)

## 📊 Features

- Configurable multi-node cluster (default: 4 nodes)
- Duplicate-aware query stream (default: 35% duplicates)
- Latency modeling for:
  - Embedding
  - Local lookup
  - Remote similarity check
  - Document retrieval
  - LLM generation
- Metrics reported:
  - Cache hit rate
  - Average latency
  - Remote similarity checks
  - Bloom filter traffic
  - False positive rate

## ⚙️ Setup

Install dependencies:

```bash
pip install simpy numpy scipy faker

## Run the script

python simulator.py
