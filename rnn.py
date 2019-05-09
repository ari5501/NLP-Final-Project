
from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import io
import time
import os
from os import path
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import numpy as np

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset


MODEL_PATH = 'rnn.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
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


#You need to write code inside functions of this class
class RNNGuesser(nn.Module):
    """.
    We use a LSTM for our guesser.
    """

    # n_input represents dimensionality of each word embedding as a vector 
    # n_output is number of answers (unique)
    def __init__(self, n_input = 10, n_hidden = 50, n_output = 2, dropout = 0.5):
        super(RNNGuesser, self).__init__()
        
        self.n_input = n_input  # size of longest question
        self.n_hidden = n_hidden
        self.n_output = n_output # amount of answers
        self.dropout = dropout

        self.lstm = nn.LSTM(self.n_input, self.n_hidden)
        self.hidden = nn.Linear(self.n_hidden, self.n_output)

        # Initalize these randomly, but we need to change them around in the foroward pass
        self.weights # figure out what these are supposed to be initialized to and 
        self.bias    # their dimensions


    # Model forward pass, returns the logits of the predictions.
    def forward(self, question_text, question_len):
        # Need a sequence of vectors as an input
        print("In the forward method")

        #get the batch size and sequence length (max length of the batch)
        batch_size, seq_len, _ = question_text.size()
        
        #Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)
        output, _ = self.lstm(question_text)
        
        #reshape (before passing to linear layer) so that each row contains one token 
        #essentially, flatten the output of LSTM 
        #dim will become batch_size*batch_max_len x lstm_hidden_dim
        reshape = output.contiguous().view(-1, output.size(2))
        
        #Get logits from the final linear layer
        logits = self.hidden_to_label(reshape)
        
        #--shape of logits -> (batch_size, seq_len, self.n_output)
        return logits


def train_model(model, checkpoint, grad_clippings, save_name, train_data_loader, dev_data_loader, accuraacy, device):
    """
    Train the current model
    Keyword arguments:
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """
    print ("starting training")
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        optimizer.zero_grad()
        out = model(question_text, question_len)
        loss = criterion(out, labels)
        loss.backward()
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


# This turns our actual questions and answers into embeddings
def get_data_info(training_data, embeddings):
    questions = training_data[0]
    answers = training_data[1]

    questions_vector = [] # 3d array to represent the question vector
    answer_vector = {} # We just map the answers to a unique index
    
    idx = 0
    for answer in answers:
        answer_vector[answer] = idx
        idx += 1

    # tokenize the question and get it into a list of words, get the embedding of each word,
    # store that embedding for the word in a list, and go on to the next part
    for question in questions:
        question_embedding = []
        tokens = question.rstrip().split(' ')
        for token in tokens:
            question_embedding.append(embeddings[token])
        question_embedding = np.asarrary(question_embedding)
        questions_vector.append(question_embedding)
    
    questions_vector = np.asarray(questions_vector) # converting to numpy array

    # Todo: Convert the 3d numpy array into a 3d tensor

    full_data = []
    full_data.append(questions_vector)
    full_data.append(answer_vector)
    
    return full_data

# Loads the embeddings if they already exist or saves them now
def load_embeddings():
    # Checking if we have already loaded the vectors before
    exists = os.path.isfile('embeddings.pickle')
    if exists:
        with open('embeddings.pickle', 'rb') as f:
            params = pickle.load(f)
            embeddings = params['fast_text_embeddings']
        print("Embeddings were loaded from embeddings.pickle")
    else:
        # Getting the embeddings from FastText
        print("Attempting to create the embeddings")
        embeddings = load_vectors("wiki-news-300d-1M.vec")
        with open('embeddings.pickle', 'wb') as f:
            pickle.dump({
                'fast_text_embeddings' : embeddings
            }, f)
        print ("Embeddings have been created and saved in embeddings.pickle")
    return embeddings


# Loads fasttext embeddings
def load_vectors(fname):
    Path(Path(os.getcwd()).parent).parent
    os.chdir('data/')
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    print(os.listdir(os.curdir))

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    index = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        index += 1
        if index % 10000 == 0:
            print ("Up to ", index, " words")
    return data


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


# This probably neeeds to be changed since we didn't write it
"""
Gather a batch of individual examples into one batch, 
which includes the question text, question length and labels 
Keyword arguments:
batch: list of outputs from vectorize function
"""
def batchify(batch):
    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


# This probably also needs to be changed since we didn't write it
def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

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




### Begin app stuff that we didn't really write ###


def create_app(enable_batch=True):
    rnn_guesser = RNNGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(rnn_guesser, question)
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

    # Ininitalize the dataset
    dataset = QuizBowlDataset(guesser_train=True)
    training_data = dataset.training_data()
    dev_data = get_dev_data(dataset)

    # Change these later, but these are random variable initializers
    checkpoint = 50
    grad_clippings = 5
    batch_size = 128
    save_name = 'rnn.pt'

    # Getting fast text embeddings
    embeddings = load_embeddings()

    # Turning the dataset data into actual vectors
    training_vectors = get_data_info(training_data, embeddings)
    dev_vectors = get_data_info(dev_data, embeddings)

    # Determining if we are using cpu or gpu
    device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Converting our training and dev data into dataloaders which work well with training our RNN
    train_sampler = torch.utils.data.sampler.RandomSampler(training_vectors, device)
    train_loader = DataLoader(training_vectors, batch_size=batch_size, sampler=train_sampler, num_workers=0,
                                           collate_fn=batchify)
    dev_sampler = torch.utils.data.sampler.RandomSampler(dev_vectors, device)
    dev_loader = DataLoader(dev_vectors, batch_size=batch_size, sampler=dev_sampler, num_workers=0,
                                           collate_fn=batchify)

    # Here we are actually running the guesser and training it
    accuracy = 0
    model = RNNGuesser()
    train_model(model, checkpoint, grad_clippings, save_name, train_loader, dev_loader, accuracy, device)
    #model.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
