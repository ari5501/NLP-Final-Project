
from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import string
import json
import io
import time
import os
from os import path
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
#import gensim
#from gensim.models import KeyedVectors

import click
import numpy as np
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from . import util
from .dataset import QuizBowlDataset


MODEL_PATH = 'rnn.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
EMB_DIM = 300


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = guess(model, [question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs


# Class for converting the tokenized data into vectors
class QuestionDataset():
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples, word2ind, num_classes, embedding, class2ind=None):
        self.questions = []
        self.labels = []
        self.class2ind = class2ind

        #print("class2ind: ", class2ind)

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)
        
        if type(self.labels[0])==str:
            class_num = 0
            for i in range(len(self.labels)):
               # print("label: ", self.labels[i])
                try:
                    self.labels[i] = self.class2ind[self.labels[i]]
                except:
                    self.labels[i] = num_classes
#                    self.class2ind[self.labels[i]] = class_num
#                    class_num += 1
        self.word2ind = word2ind
        self.embeddings = embedding
    
    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.vectorize(self.questions[index], self.embeddings, self.word2ind), \
          self.labels[index]
    
    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(ex, embeddings, word2ind):
        vec_text = [0] * len(ex)
        for i in range(0, len(ex)):
            cur_word = ex[i].lower()
            if cur_word in word2ind:
                vec_text[i] = embeddings[word2ind[cur_word]]
            else:
                # could do a bunch of zeros instead
                token_vector = np.random.normal(scale=0.6, size=(EMB_DIM, ))
                vec_text[i] = token_vector.tolist()

        return vec_text


# Guesser for our model that's implemented as an LSTM
class LSTMGuesser(nn.Module):
    # n_input represents dimensionality of each word embedding as a vector 
    # n_output is number of answers (unique)
    def __init__(self, i_to_w, w_to_i, n_input = 100, n_hidden = 128, n_output = 300, dropout = 0.3):
        super(LSTMGuesser, self).__init__()
        
        self.n_input = n_input  # size of longest question
        self.n_hidden = n_hidden # This is a hyperparameter so it could be anything
        self.n_output = n_output # amount of answers unique answers / classes
        #self.vocabsize = len(vocab)

        #self.embeddings = nn.Embedding(self.vocabsize, EMB_DIM)
        self.lstm = nn.LSTM(self.n_input, self.n_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(self.n_hidden, self.n_output) # this layer might not be needed
        print("output:", n_output)
        self.softmax = nn.Softmax()

        self.i_to_w = i_to_w
        self.w_to_i = w_to_i
        #self.vocab = vocab


    # Model forward pass, returns the logits of the predictions.
    def forward(self, question_text, question_len):
        print("In the forward method")

        #Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)
        output, _ = self.lstm(question_text)
        print("output: ", output.size()) #128 x 202 x 128

        # Pass through a dropout layer
        out = self.dropout(output) 

        print("after dropout: ", out.size()) #same dims
        
        #reshape (before passing to linear layer) so that each row contains one token 
        #essentially, flatten the output of LSTM 
        #dim will become batch_size*batch_max_len x lstm_hidden_dim
        reshape = out.contiguous().view(-1, out.size(2))
        
        print("reshape: ", reshape.size())
        #Get logits from the final linear layer
        logits = self.hidden(reshape)
        logits = logits.view(self.n_hidden, -1, 1)
        logits = logits.squeeze()
        print("logits: ", logits.size()) # should be linear 

        #--shape of logits -> (batch_size, seq_len, self.n_output)
        return logits

    # Saves the function after it is finished training so we don't have to do this all over again
    def save(self):
        with open('rnn.pickle', 'wb') as f:
            pickle.dump({
                'lstm' : self.lstm,
                'dropout' : self.dropout,
                'hidden' : self.hidden,
                'i_to_w' : self.i_to_w,
                'w_to_i' : self.w_to_i
                #'vocab' : self.vocab,
                #'embeddings' : self.embeddings
            }, f)
    
    def load(self):
        with open('rnn.pickle', 'wb') as f:
            params = pickle.load(f)
            guesser = LSTMGuesser()
            guesser.lstm = params['lstm']
            guesser.dropout = params['dropout']
            guesser.hidden = params['hidden']
            guesser.i_to_w = params['i_to_w']
            guesser.w_to_i = params['w_to_i']
            #guesser.vocab = params['vocab']
            #guesser.embeddings = params['embeddings']
            return guesser

# Get label that corresponds to the maximum logit
# This is definitely not correct
def guess(model, questions_text, max_guesses, embeddings, word2ind):

    # turn question text into feature vector - turn
    q_text = nltk.word_tokenize(questions_text)
    question_len = len(q_text)

    # Now have to vectorize q_text
    question_tensor = torch.FloatTensor(1, question_len, EMB_DIM).zero_()
    question_text = []
    for q in q_text:
        cur_word = q.lower()
        if cur_word in word2ind:
            vec_text[i] = embeddings[word2ind[cur_word]]
        else:
            token_vector = np.random.normal(scale=0.6, size=(EMB_DIM, ))
            vec_text[i] = token_vector.tolist()

    # Converting vectorized q_text to a tensor
    vec = torch.FloatTensor(question_text)
    question_tensor[1, :len(question_text)].copy_(vec)

    # put feature vector into the model - model(feature_vector, length)
    logits = model(question_tensor, question_len)

    # the logits returned would be an array of all the labels, and we want the maximum value
    top_n, top_i = logits.topk(1) # This might want to change to a larger number

    # figure out what the best label correpsonds to
    return model.i_to_w[top_i] # This probably needs to be done differently



def train_model(model, checkpoint, grad_clippings, save_name, train_data_loader, dev_data_loader, accuracy, device):
    print ("Starting training")
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    for idx, batch in enumerate(train_data_loader):
        print ("Batch ", idx)
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']
        #print(question_text.size())

        out = model(question_text, question_len)
        optimizer.zero_grad()
        #print("out: ", out)
        #print("labels: ", labels)
        loss = criterion(out, labels)
        #print("back")
        loss.backward()
        #print("optimize step")
        optimizer.step()

        clip_grad_norm_(model.parameters(), grad_clippings) 
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()

        if idx % checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, save_name)
                accuracy = curr_accuracy
    return accuracy



def batchify(batch):
    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    longest_q = max(question_len)
    # num questions, max question len, embeddings dim
    x1 = torch.FloatTensor(len(question_len), longest_q, EMB_DIM).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.FloatTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.LongTensor(question_len), 'labels': target_labels}
    print ("In the batch")
    return q_batch


# Loads the embeddings if they already exist or create and save them otherwise
def load_fast_text_embeddings():
    # Checking if we have already loaded the vectors before
    exists = os.path.isfile('embeddings.pickle')
    if exists:
        print("Attempting to load embeddings")
        with open('embeddings.pickle', 'rb') as f:
            params = pickle.load(f)
            embeddings = params['fast_text_embeddings']
        print("Embeddings were loaded from embeddings.pickle")
    else:
        # Getting the embeddings from FastText
        print("Attempting to create the embeddings")
        #embeddings = KeyedVectors.load_word2vec_format('wiki-news-300d-1m.vec', limit=300000)
        embeddings = load_vectors_wo_w2v('wiki-news-300d-1m.vec')
        with open('embeddings.pickle', 'wb') as f:
            pickle.dump({
                'fast_text_embeddings' : embeddings
            }, f)
        print ("Embeddings have been created and saved in embeddings.pickle")
    return embeddings


# Loads fasttext embeddings
def load_vectors_wo_w2v(fname):
    os.chdir('data/')
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    words = []
    index_to_word = {}
    word_to_index = {}
    index = 0

    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        words.append(word)
        index_to_word[index] = word
        word_to_index[word] = index
        vect = list(map(float, tokens[1:]))
        data.append(vect)

        index += 1
        if index % 10000 == 0:
            print(word)
            print ("Up to ", index, " words")
        if index >= 10000:
            break

    #tensor = torch.FloatTensor(data)
    return data, word_to_index, index_to_word, words


# Getting the dev data for our pytorch model
def get_dev_data(dataset):
    dev_examples = []
    dev_pages = []
    questions = []
    if dataset.guesser_train:
        questions.extend(dataset.db.guess_dev_questions)

    for q in questions:
        dev_examples.append(q.sentences)
        dev_pages.append(q.page)

    return dev_examples, dev_pages, None


# This needs to change, but Pranav said it shouldn't be too different
def evaluate(data_loader, model, device):
    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        logits = model(question_text, question_len)
        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy


# Loads the data and then tokenizes it
def load_data(filename, lim):
    files = os.listdir(os.curdir)
    dataset_path=os.path.join('data', filename)

    data = list()
    with open(dataset_path) as json_data:
        if lim>0:
            questions = json.load(json_data)["questions"][:lim]
        else:
            questions = json.load(json_data)["questions"]
        '''
        for q in questions:
            q_text = nltk.word_tokenize(q['text'])
            #label = q['category']
            label = q['page']
            if label:
                data.append((q_text, label))
        '''
        for i in range(0,1280):
            q = questions[i]
            q_text = nltk.word_tokenize(q['text'])
            label = q['page']
            if label:
                data.append((q_text, label))
    return data

# Gets the class label to index and index to class label
def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i+=1
    return class_to_i, i_to_class

# Loads the tokenized data if it exists, and creates it if it doesn't
def load_tokenized_data():
    train_data_exists = os.path.isfile('train_dataset.pickle')
    if train_data_exists:
        print ("Loading training vectors")
        with open('train_dataset.pickle', 'rb') as f:
            params = pickle.load(f)
            training_vectors = params['train_data']
    else:
        print ("Creating dev vectors")
        training_vectors = load_data('qanta.train.2018.04.18.json', -1)
        with open('train_dataset.pickle', 'wb') as f:
            pickle.dump({
                'train_data' : training_vectors
            }, f)

    dev_data_exists = os.path.isfile('dev_dataset.pickle')
    if dev_data_exists:
        print ("Loading dev vectors")
        with open('dev_dataset.pickle', 'rb') as f:
            params = pickle.load(f)
            dev_vectors = params['dev_data']
    else:
        print ("Creating dev vectors")
        dev_vectors = load_data('qanta.dev.2018.04.18.json', -1)
        with open('dev_dataset.pickle', 'wb') as f:
            pickle.dump({
                'dev_data' : dev_vectors
            }, f)

    return training_vectors, dev_vectors


def answer_to_index(train_tokenized):
    answer_to_index = {}
    index_to_answer = {}
    index = 0

    for text, answer in train_tokenized:
        answer_to_index[answer] = index
        index_to_answer[index] = answer 
        index += 1

    return answer_to_index, index_to_answer


### Begin app stuff ###


def create_app(enable_batch=True):
    rnn_guesser = LSTMGuesser.load()
    embeddings, w_to_i, i_to_w, words  = load_fast_text_embeddings()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(rnn_guesser, question, embeddings, w_to_i)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(rnn_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    # Change these later, but these are random variable initializers
    checkpoint = 50
    grad_clippings = 5
    batch_size = 128
    epochs = 5
    save_name = 'rnn.pt'


    # Getting all of the data and tokenizing it
    train_tokenized, dev_tokenized = load_tokenized_data()
    
    # Getting the number of possible answers
    num_classes = len(list(set([ex[1] for ex in train_tokenized+dev_tokenized])))

    # Getting fast text embeddings
    embeddings, w_to_i, i_to_w, words = load_fast_text_embeddings()

    # Getting the length of the longest question
    longest_q_length = max([len(ex[0]) for ex in train_tokenized+dev_tokenized])
    print ("The longest question is size" , longest_q_length)

    ans_to_i, i_to_ans = answer_to_index(train_tokenized)

    # Create the model
    model = LSTMGuesser(i_to_ans, ans_to_i, n_input = EMB_DIM, n_output = num_classes)

    # Determining if we are using cpu or gpu
    device = torch.device('cpu')

    class_to_ind, ind_to_class = class_labels(train_tokenized+dev_tokenized)

    # Converting our training and dev data into dataloaders which work well with training our RNN
    train_dataset = QuestionDataset(train_tokenized, w_to_i, num_classes, embeddings, class_to_ind)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_tokenized)


    dev_dataset = QuestionDataset(dev_tokenized, w_to_i, num_classes, embeddings, class_to_ind)
    dev_sampler = torch.utils.data.sampler.RandomSampler(dev_tokenized)
    dev_loader = DataLoader(dev_tokenized, batch_size=batch_size, sampler=dev_sampler, num_workers=0,
                                           collate_fn=batchify)

    # Here we are actually running the guesser and training it
    accuracy = 0
    print("Attmepting to train now")
    for epoch in range (epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0,
                                            collate_fn=batchify)
        accuracy = train_model(model, checkpoint, grad_clippings, save_name, train_loader, dev_loader, accuracy, device)

    # Save the model now
    print("Done training")
    model.save()



@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    #train()
    cli()
