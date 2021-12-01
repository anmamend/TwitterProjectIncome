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

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    #data_pack = input_data.reshape(1,-1)
    
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Make sure to put the model into evaluation mode
   # model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    
    #out = model(data)
    #with torch.no_grad():
     #   result_2 = model.forward(data)
        
    #result = float(np.round(result_2.numpy()))
    #result = result.cpu().detach().numpy()
    #print(result)
    
    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
    result = out.cpu().detach().numpy()

    return result
