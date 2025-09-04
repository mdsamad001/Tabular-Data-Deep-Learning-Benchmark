this folder contains experiments of attention-vs-contrastive benchmark paper.

It contains multiple baseline folder like
- tabnet/*
- NPT/*

Notebooks contain baseline methods and their experiments. They are spread into multiple files to allow running them in parallel.
- llm-paper-*.ipynb = these contain scarf experiments used as a baseline in llm benchmark paper
- baseline-*.ipynb = these contain various baselines used in the main benchmark paper. Tabnet, NPT and FTT requires using their own conda environments
- test-scarf-*.ipynb = contains contrastive learning experiments with different corruption (augmentation) methods.


helpers/* folder contain many utility files as well as the training and corruption code for contrastive learning.
- trainer*.py = contains contrastive learning training and test functions
- corruptor_df.py = contains different corruptions method used