from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import io
from os import path

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset


MODEL_PATH = 'tfidf.pickle'
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


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        # lstm vector for every word
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser

#You need to write code inside functions of this class
class RNNGuesser(nn.Module):
    """.
    We use a LSTM for our guesser.
    """

    #n_input represents dimensionality of each word embedding as a vector 
    #n_output is number of answers (unique)
    def __init__(self):
        super(RNNGuesser, self).__init__()
        
        self.n_input = 10  # size of longest question
        self.n_hidden = 50 
        self.n_output = 2 # amount of answer
        self.dropout = 0.5

    def train(self, training_data) -> None:
        # lstm vector for every word in question

        # given a question
        # turn every word in the question into a vector using embedding
        # give that sequence of word vectors to the LSTM to train with
        # collect unique answers, set that to classification #s
        # train the lstm to read question vector and return answer
        # lstm returns final hidden layer, apply softmax to get answer

        #READ ALL THE QUESTIONS AND ANSWERS
        questions = training_data[0]
        answers = training_data[1]

        self.n_input = len(questions) # max sized question
        self.n_output = len(answers)

        # TEXT = questions joined by " "
        # answer_docs[each answer] += question text
        # this generates a bag of words for every unique answer
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        # creating embeddings of all words in wikipedia using Facebook's FastText
        embeddings = load_vectors("../data/wiki-news-300d-1M.vec")

        
        #define lstm layer, going from input to hidden. Remember to have batch_first=True.
        self.lstm = nn.LSTM(self.n_input, self.n_hidden, batch_first=True)
        
        #define linear layer going from hidden to output.
        self.hidden_to_label = nn.Linear(self.n_hidden, self.n_output)
        

        # Have to actually train the data now


    def forward(self, X, X_lens):
        
        #Model forward pass, returns the logits of the predictions.
        
        #Keyword arguments:
        #input_text : vectorized question text 
        #text_len : batch * 1, text length for each question
        #is_prob: if True, output the softmax of last layer
        

        #get the batch size and sequence length (max length of the batch)
        #dim of X: batch_size x batch_max_len x input feature vec dim
        batch_size, seq_len, _ = X.size()
        
        ###Your code here --
        #Get the output of LSTM - (output dim: batch_size x batch_max_len x lstm_hidden_dim)
        output, _ = self.lstm(X)
        
        
        #reshape (before passing to linear layer) so that each row contains one token 
        #essentially, flatten the output of LSTM 
        #dim will become batch_size*batch_max_len x lstm_hidden_dim
        reshape = output.contiguous().view(-1, output.size(2))
        

        
        #Get logits from the final linear layer
        logits = self.hidden_to_label(reshape)
        
        
        #--shape of logits -> (batch_size, seq_len, self.n_output)
        return logits

# Loads fasttext embeddings
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def create_app(enable_batch=True):
    tfidf_guesser = TfidfGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
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
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, questions)
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
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()


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
