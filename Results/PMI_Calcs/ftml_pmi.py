import io
from math import log
import os
import pickle
from numpy import array
from numpy import argmax
import torch
import random
from math import log
from numpy import array
from numpy import argmax
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torchcrf import CRF
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from typing import List, Tuple, AnyStr
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import load_dataset, load_metric
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import transformers
from transformers import AutoTokenizer, AdamW
from transformers import TrainingArguments, Trainer
import transformers
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from datasets import DatasetDict
from dataclasses import dataclass
import random
import time
import datetime
import sys
import math


def enforce_reproducibility(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
enforce_reproducibility()

HG_MODEL_NAME = "ernlavr/destilbert_uncased_fever_nli"
HG_DATASET = "pietrolesci/nli_fever"
NUM_LABELS = 3

def loadTokenizer():
    return AutoTokenizer.from_pretrained(HG_MODEL_NAME)

def loadFeverDataset():
    return load_dataset(HG_DATASET)

def idxToLabels():
    return {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}

def balance_dataset(ds, numSamples=-1):
    """
    Balances the dataset by removing samples from the majority class
    :param ds: The dataset
    :param numSamples: The number of samples to keep
    :return: The balanced dataset
    """
    # Get the number of samples for each label
    dss = ds[:]
    labels = dss["fever_gold_label"]
    if numSamples == -1:
        numSamples = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        counts = np.roll(counts, 1)
        unique = np.roll(unique, 1)
        numSamples = min(counts)

    # get indices of ds elements where ds['label'] is 0
    arr = dss['label']
    arr = np.array(arr)
    indicesSup = np.where(arr == 0)[0][:numSamples]
    indicesNei = np.where(arr == 1)[0][:numSamples]
    indicesRef = np.where(arr == 2)[0][:numSamples]

    # combine the indices
    indices = np.sort((np.concatenate((indicesSup, indicesNei, indicesRef))))
    indices = indices.tolist()
    # get a subset of the dataset
    return indices
    
def tokenize_function(examples):
    textPairs = zip(examples["premise"], examples["hypothesis"])
    textPairs = [pair[0] + " " + pair[1] for pair in textPairs]
    out = tokenizer(textPairs, padding="max_length", truncation=True)
    out.data["label"] = examples["label"]
    return out

from collections import defaultdict
import math
from dataclasses import dataclass

@dataclass
class PMI_data:
    def __init__(self):
        self.freq_x = defaultdict(int)
        self.freq_y = defaultdict(int)
        self.freq_xy = defaultdict(int)
        self.n = 0

def getPmiData(dataset : torch.utils.data.dataset.Subset, tokenizer : PreTrainedTokenizerFast):
    # Initialize dictionaries to store the count of co-occurrences
    pmiData = PMI_data()

    # Iterate through the dataset to gather counts
    for example in tqdm(dataset):
        label = example['label']
        tokens = tokenizer.convert_ids_to_tokens(example['input_ids'], True)
        for token in tokens:
            pmiData.freq_x[token] += 1
            pmiData.freq_y[label] += 1
            pmiData.freq_xy[(token, label)] += 1
            pmiData.n += 1
    with open(pmiPath, 'wb') as handle:
        pickle.dump(pmiData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pmiData

def calculate_pmis(pmiData : PMI_data):
    """ Calculates the pointwise mutual information for each token and corresponding label of the input """
    
    freq_x = pmiData.freq_x
    freq_y = pmiData.freq_y
    freq_xy = pmiData.freq_xy
    n = pmiData.n
    
    # Calculate the PMI
    datasetPMIs = defaultdict(dict)
    for token in freq_x:
        for label in freq_y:
            x = freq_x[token]
            y = freq_y[label]
            xy = freq_xy[(token, label)]
            if xy == 0:
                datasetPMIs[token][label] = 0
            else:
                datasetPMIs[token][label] = math.log2((xy * n) / (x * y))
    
    return datasetPMIs

def print_pmis(datasetPMIs, tt=15):
    # get the top 10 tokens for each label
    findPmis = ["nothing", "segments", "recipe", "cried", "summarized", "emia"]
    topTokens = defaultdict(list)
    for token in datasetPMIs:
        for label in datasetPMIs[token]:
            topTokens[label].append((token, datasetPMIs[token][label]))
    for label in topTokens:
        topTokens[label] = sorted(topTokens[label], key=lambda x: x[1], reverse=True)[:tt]


    # Print the top 10 tokens for each label
    for label in sorted(topTokens):
        print(f"Top 10 tokens for label {label}")
        for token in topTokens[label]:
            print(f"\t{token[0]} {round(token[1], 4)}")
    
    # for each token compute the highest absolute difference between the PMI for the three labels and save them in a list
    diffs = []
    for token in datasetPMIs:
        lbldiffs = []
        for label in datasetPMIs[token]:
            lbldiffs.append(datasetPMIs[token][label])
        maxDiff = max(datasetPMIs[token].values()) - min(datasetPMIs[token].values())
        diffs.append((token, maxDiff, lbldiffs))

    # sort diffs simulataneously by the max difference and the sum of the PMIs
    diffs = sorted(diffs, key=lambda x: (x[1], x[2]), reverse=True)


    # Print tokens from findPmis
    print("Tokens from findPmis")
    for token in datasetPMIs:
        if token in findPmis:
            print(f"\t{token} {datasetPMIs[token]}")


    return datasetPMIs


# Load and parse the dataset
ds = loadFeverDataset()
tokenizer = loadTokenizer()

dsSplit = "train"
dsSize = 7500 # train 7500, dev 450
pmiPath = f"datasetPMIdata_{dsSplit}.pickle"
pmisName = f"datasetPMIs_{dsSplit}.pickle"

# Map the dataset to the tokenizer
devIdx = balance_dataset(ds[dsSplit], dsSize)
tds = ds.map(tokenize_function, batched=True)
devSet = torch.utils.data.Subset(tds[dsSplit], devIdx)

# Fetch the caches
pmiData = None
pmis = None
if os.path.exists(pmiPath):
    with open(pmiPath, 'rb') as handle:
        pmiData = pickle.load(handle)
else:
    pmiData = getPmiData(devSet, tokenizer)

# Calc and print
pmis = calculate_pmis(pmiData)
print_pmis(pmis)



