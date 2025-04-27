# CAFA5 Challenge Solution

### This is still work in progress

For learning purposes I wanted to try to fine-tune a large language model using "my own" data (not my own but I guess some data). 

Coming from bioinformatics I thought protein function prediction would be a nice application for this.

So I decided make my own solution for the [CAFA5 Challenge](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/overview).

In my approach I want to use PEFT (specifially LoRA) to fine-tune a protein language model (PLM) (specifically [ESM2]()) on the competition data and use the embeddings generated from the PLM and use them for predicting protein function.

To compare the preformance of fine-tuning on this dataset will be measured against a baseline by taking embeddings generated from the pretrained model.

### Roadmap

- [x] Get the embeddings for the protein sequences in the training set (esm2_8M and esm2_650M)
- [x] Fine-tune ESM Model
- [ ] Train protein function classifier
- [ ] Submit solution for the challenge and get final evalutation
