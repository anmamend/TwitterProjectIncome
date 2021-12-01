import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from bs4 import BeautifulSoup
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re


from io import StringIO
from six import BytesIO

from model import LSTMClassifier
CONTENT_TYPE = 'application/x-npy'

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stopwords2 =set(stopwords.words('spanish')+['colombia', 'bogotá', 'cartagena', 'medellín',
                                          'barranquilla', 'cali', 'antioquia', 'co', 'just', 'posted',
                                          'medellin','bogota', 'si', 'así', 'dio'])
    spanish_stemmer = SnowballStemmer('spanish')
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
               '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)
    text = re.sub("(#[A-Za-zá-úÁ-Ú0-9_]+)",'', text)
    text = re.sub("(@[A-Za-zá-úÁ-Ú0-9_]+)","", text)
    text = re.sub(r"[^a-zA-Zá-úÁ-Ú0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords2] # Remove stopwords
    words = [spanish_stemmer.stem(word) for word in  words] # stem
    
    return words

def convert_and_pad(word_dict, list_sentence, pad=35):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(list_sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(list_sentence), pad)


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if (content_type == 'text/plain') | (content_type== 'application/octet-stream'):
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    #data_pack = input_data.reshape(1,-1)
    input_data_reviewed=review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict,input_data_reviewed)
    np.hstack((data_len, data_X))
    
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)

    data = torch.from_numpy(data_pack)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    with torch.no_grad():
        result_2 = model.forward(data)
        
    result = float(result_2.numpy())

    return result
