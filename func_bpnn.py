# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot
import time


class KeyWord(object):
    def __init__(self, pathX, pathY):
        self.pathX = pathX
        self.pathY = pathY

    def initialize(self):
        # X data preprocessing#####################
        soup = BeautifulSoup(open(self.pathX, 'rb'), 'html.parser')
        text = soup.get_text().split('\n')    # extract plain text from html
        joblist = []               # initialize joblist
        ctext, jobid = '', ''
        for line in text:  # loop through whole text by lines
            A = len(line) > 1
            B = re.search('^####.*', line)
            C = re.search('^----.*', line)
            if A and not B and not C:
                if re.search('^Job Id: .*', line):  # if start with 'Job Id: '
                    # append cjob to joblist if cjob is not empty
                    if jobid != '':
                        joblist.append([[jobid], [ctext]])
                        ctext = ''
                    jobid = str(re.findall('Job Id: (.*)', line))[2:-2]
                    continue
                cline = line.lower()             # set string to lower case
                # split paragraph to words
                words = re.split(' |, |; |\. |: |\.$|:$|/|/ |\(|\)|\?|\!|\:',
                                 cline)
                if len(words) > 0:
                    ctext += ''.join(str(word)+' ' for word in words)+'\n'

        self.Xid = pd.DataFrame(np.array(joblist).reshape(len(joblist), 2),
                                columns=('ID', 'Content')).set_index('ID')

        # Y data preprocessing#######################

        Y = pd.read_csv(self.pathY, delimiter='\t', header=None)
        # Flatten keyword list
        for i in range(len(Y)):
            flat = ''.join(
                str(word)+' ' for word in re.split('\,|\|', Y.iloc[i, 1])
                )
            Y.iloc[i, 1] = flat
        Y.columns = ('ID', 'Content')
        self.Yid = Y.set_index('ID')
        print('Data Loaded.')

    # data splitting#################################
    def datasplit(self, s=0.2):
        # randomly 8/2 split Ytrain into train and valid
        self.Ytrain, self.Yvalid = train_test_split(self.Yid, test_size=s)
        # split X accordingly
        self.Xtrain = self.Xid.reindex(self.Ytrain.index.values)
        self.Xvalid = self.Xid.reindex(self.Yvalid.index.values)
        self.Xtest = self.Xid.reindex(
            self.Xid.index.difference(self.Yid.index)
            )
        self.Xtest0 = self.Xtest.copy()
        print('Train-valid data splitted.')

    # tokenizing###########################
    # tokenize by space
    def _token_s(self, df):
        tokened = []
        for i in range(len(df)):
            tokened.append(str(df.iloc[i, 0]).lower().split())
        return tokened

    def tokenize(self):
        # tokenize all JD
        self.Xt = self._token_s(self.Xid)
        # tokenize Xtrain,Ytrain
        self.Xtrain = self._token_s(self.Xtrain)
        self.Ytrain = self._token_s(self.Ytrain)
        # tokenize Xvalid,Yvalid
        self.Xvalid = self._token_s(self.Xvalid)
        self.Yvalid = self._token_s(self.Yvalid)
        # tokenize Xtest
        self.Xtest = self._token_s(self.Xtest)
        # del self.Xid, self.Yid
        print('Data tokenized.')

    # training word2vec model###############
    def w2v(self, w2vsize=128):
        self.w2vsize = w2vsize
        self.model_t2v = Word2Vec(self.Xt, min_count=1, size=self.w2vsize)
        # show top ten most similar words
        print('Top10 most similar words to \'Python\':')
        top10 = self.model_t2v.wv.most_similar('python')
        for i in top10:
            print(i)
        # model_t2v.wv.get_vector('python') #show word vector

    # vectorizing###########################
    # extract unique words from nestedlist(token)
    def _uni_word(self, nestedlist):
        # flatten nested list
        flat_list = [item for items in nestedlist for item in items]
        return list(set(flat_list))

    def unique(self):
        # train unique words
        self.Xtrainu = self._uni_word(self.Xtrain)
        self.Ytrainu = self._uni_word(self.Ytrain)
        # valid unique words
        self.Xvalidu = self._uni_word(self.Xvalid)
        self.Yvalidu = self._uni_word(self.Yvalid)
        # test unique words
        self.Xtestu = self._uni_word(self.Xtest)
        print('Unique words picked.')

    # vectorize text data
    def _v0(self, X, Y):
        Xv = np.zeros((len(X), self.w2vsize))  # initialize Xtrainv vector
        Yv = np.zeros(len(X))             # initialize Ytrainv vector
        unknown = []
        # loop through train sample size(trainN)
        for i in range(len(X)):
            try:
                # create Xtrainv vector
                Xv[i, :] = self.model_t2v.wv.get_vector(X[i])
                # create Ytrainv vector
                if X[i] in Y:
                    Yv[i] = 1
            # record potential words not in word2vec dict
            except KeyError:
                unknown.append(X[i])
        # check word capture rate (expected to be 1?)
        print('Word capture rate: {:.4f}'.format(Yv.sum() / len(Y)))
        return Xv, Yv

    # vectorize text data
    def _v1(self, X):
        # initialize Xtrainv vector
        Xv = np.zeros((len(X), self.w2vsize))
        unknown = []
        # loop through train sample size(trainN)
        for i in range(len(X)):
            try:
                # create Xtrainv vector
                Xv[i, :] = self.model_t2v.wv.get_vector(X[i])
            except KeyError:
                # record potential words not in word2vec dict
                unknown.append(X[i])
        return Xv

    # neural network###############
    # Tensorize data
    def tensorize(self):
        # Vertorize
        self.Xtrain, self.Ytrain = self._v0(self.Xtrainu, self.Ytrainu)
        self.Xvalid, self.Yvalid = self._v0(self.Xvalidu, self.Yvalidu)
        self.Xtest = self._v1(self.Xtestu)
        # Tensorize
        self.Xtrain = torch.tensor(scale(self.Xtrain)).float()
        self.Ytrain = torch.tensor(self.Ytrain).long()
        self.Xvalid = torch.tensor(scale(self.Xvalid)).float()
        self.Yvalid = torch.tensor(self.Yvalid).long()
        self.Xtest = torch.tensor(scale(self.Xtest)).float()
        print('Words tensorized.')

    # Train model
    def _bpnn(self, opt):
        D = torch.utils.data.TensorDataset(self.Xtrain, self.Ytrain)
        # set batches
        loader = torch.utils.data.DataLoader(dataset=D, batch_size=self.mbs)
        # set neural network
        model = nn.Sequential(nn.Linear(self.w2vsize, self.hls),
                              nn.ReLU(),
                              nn.Linear(self.hls, 2)
                              )
        # set optimizer and loss function
        optimizer = opt(model.parameters())
        lossf = nn.CrossEntropyLoss()
        loss_accu = np.zeros((self.epo, 3))
        tic = time.time()
        for epoch in range(self.epo):
            acculist = []     # training errors
            for step, (batch_x, batch_y) in enumerate(loader):
                # Forward Propagation ###########
                model.train()
                y_predtrain = model(batch_x)
                # Compute loss
                loss = lossf(y_predtrain, batch_y)

                # Backward Propagation ###########
                # Zero the gradients
                optimizer.zero_grad()
                # perform a backward pass (backpropagation)
                loss.backward()
                # Update the parameters
                optimizer.step()

                # (-1,1)->(0,1) predlabels
                ypred = (np.sign(y_predtrain.detach().numpy())+1)/2
                ylabel = batch_y.detach().numpy()  # true labels
                accu = accuracy_score(ypred[:, 1], ylabel)
                acculist.append(accu)

                model.eval()
                ypred = model(self.Xvalid)
                lossv = lossf(ypred, self.Yvalid)

                # print current k and epoch per epoch
            loss_accu[epoch] = np.array([loss.item(),
                                         np.mean(acculist),
                                         lossv.item()])
            if epoch % 10 == 9:
                toc = time.time()
                print('epoch:{0:3d}/{1}    '
                      'loss_train:{2:.4f}    '
                      'loss_valid:{3:.4f}    '
                      'accu:{4:.4f}    '
                      'elapsed:{5:.2f}'
                      .format(epoch+1, self.epo, loss.item(), lossv.item(),
                              np.mean(acculist), toc-tic))
        return model, loss_accu

    def model_train(self, mbs=128, hls=256, epo=500, opt=torch.optim.Adadelta):
        self.mbs = mbs    # minibatch size
        self.hls = hls    # hidden layer size
        self.epo = epo    # epoch size
        self.model_nn, self.loss_accu = self._bpnn(opt)
        print('Neuron network trained.')

    # plot the graph ##############
    def graphplot(self):
        pyplot.clf()
        pyplot.figure(1)
        pyplot.plot(self.loss_accu[:, 0], 'r-', label='train loss')
        pyplot.plot(self.loss_accu[:, 2], 'b-', label='valid loss')
        pyplot.legend(loc='best', shadow=True, fontsize=13)
        pyplot.title('NN with 1 hidden layer of 256 neurons', fontsize=18)
        pyplot.xlabel('Epoch', fontsize=17)
        pyplot.ylabel('Loss', fontsize=17)
        pyplot.figure(2)
        pyplot.plot(self.loss_accu[:, 1], 'b-', label='train accuracy')
        pyplot.legend(loc='best', shadow=True, fontsize=13)
        pyplot.title('NN with 1 hidden layer of 256 neurons', fontsize=18)
        pyplot.xlabel('Epoch', fontsize=17)
        pyplot.ylabel('Accuracy', fontsize=17)
        pyplot.show()

    # Fitting data ################
    def _bpnnfit(self, model, X):
        model.eval()
        y_pred = model(X)
        y_pred = (np.sign(y_pred.detach().numpy())+1)/2
        return y_pred

    def _check(self, Y, Yp):
        Y = Y.detach().numpy()
        accu = accuracy_score(Yp[:, 1], Y)
        return accu

    def _devect(self, y_pred, Xu):
        Y_ind = np.nonzero(y_pred[:, 1])[0]   # get index of predicted skills
        Y_skill = [Xu[i] for i in Y_ind]   # devectorize
        return Y_skill

    def model_fit(self):
        # fit train data
        self.y_predtrain = self._bpnnfit(self.model_nn, self.Xtrain)
        self.accutrain = self._check(self.Ytrain, self.y_predtrain)
        # fit valid data
        self.y_predvalid = self._bpnnfit(self.model_nn, self.Xvalid)
        self.accuvalid = self._check(self.Yvalid, self.y_predvalid)
        # fit test data
        self.y_predtest = self._bpnnfit(self.model_nn, self.Xtest)

        # devectorize
        self.y_predtrain = self._devect(self.y_predtrain, self.Xtrainu)
        self.y_predvalid = self._devect(self.y_predvalid, self.Xvalidu)
        self.y_predtest = self._devect(self.y_predtest, self.Xtestu)
        print('Data fitted.')

    # doing Assignment 1 to find associated skills
    def _asgn1(self, Y_testskill, Xtest0):
        keywords = Y_testskill.copy()   # get predicted keywords
        output = Xtest0.copy()   # output df
        for i in range(len(Xtest0)):   # loop through valid seti=0
            cjob = []   # initialize
            textlist = Xtest0.iloc[i][0].split()  # current JD
            for key in keywords:  # loop through keywords
                if key in textlist:
                    cjob.append(key)  # check by keys
            output.iloc[i][0] = cjob
        output = output.reset_index(drop=False)
        # np.savetxt(r'FinalProject.txt',output.values, fmt='%s')
        return output

    def keymatch(self):
        self.output = self._asgn1(self.y_predtest, self.Xtest0)
        print('Keywords matched.')

    # alternative assessment##############
    def _pr(self, Y_skill, Yu, label):
        correct = 0
        for i in Y_skill:
            if i in Yu:
                correct += 1
        pre = correct / len(Y_skill)
        rec = correct / len(Yu)
        print('\nFor {}, {} captured out of {}, {} are corrected.\n'
              .format(label, len(Y_skill), len(Yu), correct),
              'Precision: {:.6f}.'.format(pre),
              'Recall: {:.6f}.\n'.format(rec),
              'F1 score: {:.6f}.\n'.format(2*pre*rec/(pre+rec)))

    def f1score(self):
        self._pr(self.y_predtrain, self.Ytrainu, 'Train')
        self._pr(self.y_predvalid, self.Yvalidu, 'Valid')
