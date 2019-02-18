import time
import random
import pandas as pd
import numpy as np
import gc
import re
import torch
from torchtext import data
import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter
from textblob import TextBlob
from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchtext.data import Example
from sklearn.metrics import f1_score
import torchtext
import os 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer
from unidecode import unidecode

from model import Embed_Layer,GRU_Layer,Caps_Layer,Capsule_Main,Attention,NeuralNet
from data.load import load_glove,load_fasttext,load_para
from data.preprocess import build_cvocab,add_features,MyDataset
from Cyclic CLR import CyclicLR





def train():
    for i, (train_idx, valid_idx) in enumerate(splits):    
        # split data in train / validation according to the KFold indeces
        # also, convert them to a torch tensor and store them on the GPU (done with .cuda())
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        features = np.array(features)

        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

        kfold_X_features = features[train_idx.astype(int)]
        kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

    #     model = BiLSTM(lstm_layer=2,hidden_dim=40,dropout=DROPOUT).cuda()
        model = NeuralNet()

        # make sure everything in the model is running on the GPU
        model.cuda()

        # define binary cross entropy loss
        # note that the model returns logit to take advantage of the log-sum-exp trick 
        # for numerical stability in the loss
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        step_size = 300
        base_lr, max_lr = 0.001, 0.003   
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=max_lr)

        ################################################################################################
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                   step_size=step_size, mode='exp_range',
                   gamma=0.99994)
        ###############################################################################################

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train = MyDataset(train)
        valid = MyDataset(valid)

        ##No need to shuffle the data again here. Shuffling happens when splitting for kfolds.
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
        for epoch in range(n_epochs):
            # set train mode of the model. This enables operations which are only applied during training like dropout
            start_time = time.time()
            model.train()

            avg_loss = 0.  
            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                # Forward pass: compute predicted y by passing x to the model.
                ################################################################################################            
                f = kfold_X_features[index]
                y_pred = model([x_batch,f])
                ################################################################################################

                ################################################################################################

                if scheduler:
                    scheduler.batch_step()
                ################################################################################################


                # Compute and print loss.
                loss = loss_fn(y_pred, y_batch)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the Tensors it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
            model.eval()

            # predict all the samples in y_val_fold batch per batch
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((len(df_test)))

            avg_val_loss = 0.
            for i, (x_batch, y_batch, index) in enumerate(valid_loader):
                f = kfold_X_valid_features[index]
                y_pred = model([x_batch,f]).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss) 
        # predict all samples in the test set batch per batch
        for i, (x_batch,) in enumerate(test_loader):
            f = test_features[i * batch_size:(i+1) * batch_size]
            y_pred = model([x_batch,f]).detach()

            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))


if __name__ == '__main__':
    x_train, x_test, y_train, features, test_features, word_index = load_and_prec() 
    glove_embeddings = load_glove(word_index)
    paragram_embeddings = load_para(word_index)

    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
    train()
    




   
