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


# X data preprocessing#########################################################
def datainX(path):
    soup = BeautifulSoup(
        open(path, 'rb'), 'html.parser'
        )
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
    return joblist


joblist = datainX("allJobDescription.html")
Xid = pd.DataFrame(np.array(joblist).reshape(len(joblist), 2),
                   columns=('ID', 'Content')).set_index('ID')
del joblist


# Y data preprocessing########################################################
def datainY(path):
    Y = pd.read_csv(path, delimiter='\t', header=None)
    # Flatten keyword list
    for i in range(len(Y)):
        flat = ''.join(
            str(word)+' ' for word in re.split('\,|\|', Y.iloc[i, 1])
            )
        Y.iloc[i, 1] = flat
    Y.columns = ('ID', 'Content')
    return Y


Yid = datainY("job_skills_training.txt").set_index('ID')


# data splitting##############################################################
# randomly 8/2 split Ytrain into train and valid
Ytrain, Yvalid = train_test_split(Yid, test_size=0.2)
# split X accordingly
Xtrain = Xid.reindex(Ytrain.index.values)
Xvalid = Xid.reindex(Yvalid.index.values)
Xtest = Xid.reindex(Xid.index.difference(Yid.index))
Xtest0 = Xtest.copy()


# tokenizing##################################################################
# tokenize by space
def token_s(df):
    tokened = []
    for i in range(len(df)):
        tokened.append(str(df.iloc[i, 0]).lower().split())
    return tokened


# tokenize all JD
Xt = token_s(Xid)
# tokenize Xtrain,Ytrain
Xtrain = token_s(Xtrain)
Ytrain = token_s(Ytrain)
# tokenize Xvalid,Yvalid
Xvalid = token_s(Xvalid)
Yvalid = token_s(Yvalid)
# tokenize Xtest
Xtest = token_s(Xtest)
del Xid, Yid

# training word2vec model#####################################################
w2vsize = 128
model_t2v = Word2Vec(Xt, min_count=1, size=w2vsize)
# show top ten most similar words
model_t2v.wv.most_similar('python')
# model_t2v.wv.get_vector('python') #show word vector


# vectorizing#################################################################
# extract unique words from nestedlist(token)
def uni_word(nestedlist):
    # flatten nested list
    flat_list = [item for items in nestedlist for item in items]
    return list(set(flat_list))


# train unique words
Xtrainu = uni_word(Xtrain)
Ytrainu = uni_word(Ytrain)
# valid unique words
Xvalidu = uni_word(Xvalid)
Yvalidu = uni_word(Yvalid)
# test unique words
Xtestu = uni_word(Xtest)


# vectorize text data
def vectorize_0(X, Y, w2vsize, model_t2v):
    Xv = np.zeros((len(X), w2vsize))  # initialize Xtrainv vector
    Yv = np.zeros(len(X))             # initialize Ytrainv vector
    unknown = []
    # loop through train sample size(trainN)
    for i in range(len(X)):
        try:
            # create Xtrainv vector
            Xv[i, :] = model_t2v.wv.get_vector(X[i])
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
def vectorize_1(X, w2vsize, model_t2v):
    # initialize Xtrainv vector
    Xv = np.zeros((len(X), w2vsize))
    unknown = []
    # loop through train sample size(trainN)
    for i in range(len(X)):
        try:
            # create Xtrainv vector
            Xv[i, :] = model_t2v.wv.get_vector(X[i])
        except KeyError:
            # record potential words not in word2vec dict
            unknown.append(X[i])
    return Xv


Xtrain, Ytrain = vectorize_0(Xtrainu, Ytrainu, w2vsize, model_t2v)
Xvalid, Yvalid = vectorize_0(Xvalidu, Yvalidu, w2vsize, model_t2v)
Xtest = vectorize_1(Xtestu, w2vsize, model_t2v)


# neural network#############################################################
# Tensorize data
Xtrain = torch.tensor(scale(Xtrain)).float()
Ytrain = torch.tensor(Ytrain).long()
Xvalid = torch.tensor(scale(Xvalid)).float()
Yvalid = torch.tensor(Yvalid).long()
Xtest = torch.tensor(scale(Xtest)).float()

Xtrain = Xtrain.to(device=torch.device('cuda'))
Ytrain = Ytrain.to(device=torch.device('cuda'))


# Train model
def bpnn(X, Y, mbs, w2vs, hls, epo):
    dataset = torch.utils.data.TensorDataset(X, Y)
    # set batches
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=mbs)
    # set neural network
    model = nn.Sequential(nn.Linear(w2vs, hls),   # input to hidden
                          nn.ReLU(),              # activation for hidden
                          nn.Linear(hls, 2))      # hidden to output
    model.cuda()
    # set optimizer and loss function
    optimizer = torch.optim.Adadelta(model.parameters())
    lossf = nn.CrossEntropyLoss()
    loss_accu = np.zeros((epo, 2))
    tic = time.time()

    for epoch in range(epo):
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
            ypred = (np.sign(y_predtrain.cpu().detach().numpy())+1)/2
            ylabel = batch_y.cpu().detach().numpy()  # true labels
            accu = accuracy_score(ypred[:, 1], ylabel)
            acculist.append(accu)
        # print current k and epoch per epoch
        loss_accu[epoch] = np.array([loss.cpu().item(), np.mean(acculist)])
        if epoch % 10 == 9:
            toc = time.time()
            print('epoch:{0}/{1}    loss:{2:.4f}    '
                  'accu:{3:.4f}    elapsed:{4:.2f}'
                  .format(epoch+1, epo, loss.cpu().item(),
                          np.mean(acculist), toc-tic))
            tic = time.time()
    return model, loss_accu


mbs = 128    # minibatch size
hls = 256    # hidden layer size
epo = 100    # epoch size
model0, loss_accu = bpnn(Xtrain, Ytrain, mbs, w2vsize, hls, epo)


# plot the graph ##############
def graphplot(loss_accu):
    pyplot.clf()
    pyplot.figure(1)
    pyplot.plot(loss_accu[:, 0], 'r-', label='train loss')
    pyplot.legend(loc='best', shadow=True, fontsize=17)
    pyplot.title('NN with 1 hidden layer of 256 neurons', fontsize=18)
    pyplot.xlabel('epoch', fontsize=17)
    pyplot.ylabel('train loss', fontsize=17)
    pyplot.figure(2)
    pyplot.plot(loss_accu[:, 1], 'b-', label='train accuracy')
    pyplot.legend(loc='best', shadow=True, fontsize=17)
    pyplot.title('NN with 1 hidden layer of 256 neurons', fontsize=18)
    pyplot.xlabel('epoch', fontsize=17)
    pyplot.ylabel('train accuracy', fontsize=17)
    pyplot.show()


graphplot(loss_accu)


# Fitting data ################
def bpnnfit(model, X):
    model.eval()
    y_pred = model(X)
    y_pred = (np.sign(y_pred.detach().numpy())+1)/2
    return y_pred


def check(Y, Yp):
    Y = Y.detach().numpy()
    accu = accuracy_score(Yp[:, 1], Y)
    return accu


model0.cpu()
Xtrain = Xtrain.to(device=torch.device('cpu'))
Ytrain = Ytrain.to(device=torch.device('cpu'))

# fit train data
y_predtrain = bpnnfit(model0, Xtrain)
accutrain = check(Ytrain, y_predtrain)

# fit valid data
y_predvalid = bpnnfit(model0, Xvalid)
accuvalid = check(Yvalid, y_predvalid)

# fit test data
y_predtest = bpnnfit(model0, Xtest)


# devectorizing###############################################################
def devect(y_pred, Xu):
    Y_ind = np.nonzero(y_pred[:, 1])[0]   # get index of predicted skills
    Y_skill = [Xu[i] for i in Y_ind]   # devectorize
    return Y_skill


Y_trainskill = devect(y_predtrain, Xtrainu)
Y_validskill = devect(y_predvalid, Xvalidu)
Y_testskill = devect(y_predtest, Xtestu)


# doing Assignment 1 to find associated skills
def asgn1(Y_testskill, Xtest0):
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
    # np.savetxt(r'D:\5934Assign1\FinalProject.txt', output.values, fmt='%s')
    return output


output = asgn1(Y_testskill, Xtest0)


# alternative assessment######################################################
def pr(Y_skill, Yu, label):
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


pr(Y_trainskill, Ytrainu, 'Train')
pr(Y_validskill, Yvalidu, 'Valid')
