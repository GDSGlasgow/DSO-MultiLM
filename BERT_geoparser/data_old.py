#!/usr/bin/env python3
# standard library imports
import sys
import os
import re
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party imports
import pathlib
import numpy as np
import pandas as pd
from sklearn import preprocessing
# local imports
from BERT_geoparser.tokenizer import Tokenizer
from BERT_geoparser.utils import flatten

class Data:
    """An object for handling data and passing it into a BERT model. 

    Attributes
    ----------
    tokenizer_obj : tokenizer.Tokenizer 
        A tokenizer object.
    max_len : int
        The maximum number of tokens per input. This will be determined by the 
        setup, and will be constrained by the hardware used to train the model. 
    data_path : pathlib.Path
        Path to the csv data.
    label_encoder : preprocessing.LabelEncoder
        Used to encode labeld (from ids -> tags).
    sentences : list
        List of list of tokens in each sentence in the data
    tags : list
        List of list of tags for each token in each sentence in the data.
    tag_dict : dict
        Lookup dictionary between ids and tags.

    Methods
    -------
    process_csv() -> list, list:
        Returns lists of [0] the tokens in each sentence
        (e.g. [[S0T0, S0T1,...,S0TN],[S1T0,...,S1TN],...,[SMT0,...,SMTN]) for
        a datset with M sentences and a N tokens per sentence; and [1] the tags
        given to each token in each sentence (simil;ar structure to [0]).
    build_train_data() -> np.array, np.array:
        Bulds an X and y input for a BERT model based on the sentences and tags 
        produced by process_csv().
    build_input_from_text(text:str) -> list:
        Takes a sentence string, tokenizes and turns into valid X input for the 
        BERT model.
    encode_sentence(sentence:list, tag:list) -> list, list:
        Takes a list of tokens for a sentence and the corresponding tags and
        encodes as ids (numeric versions of tokens) and adds [CLS]/[SEP] ids
        and tags to identify start and end of sentence. 
    add_special_tokens(ids:list, tags:list) -> list, list:
        Add the [CLS]/[SEP] ids and tags to the provided ids and tags.
    add_padding(ids:list, tags:list) -> list, list
        Adds the required padding to a list of ids and tags such that the lenght
        of both is equal to the specified max_len attribute.
    get_tag_dict() -> dict:
        Reutrns a dictionary mapping tag names ([B-geo], [I-per] etc) to the
        numeric values used by the model. 
    """

    def __init__(self, data_path:str, tokenizer:Tokenizer, max_len:int):
        """Initializes the data object
        parameters
        ----------
        data_path : str
            Path to the csv data being use. The data should have columns 
            'Sentence #', 'Word', and 'Tag'. 
        tokenizer : tokenizer.Tokenizer
            The tokenizer object used to produce input tokens. 
        max_len : int
            The maximum number of tokens per sentence. This is determined by the
            modle configuration and will depend on the memory available when 
            training the model. max_len = 125 works for an 8gb GPU. 
        """
        self.tokenizer_obj = tokenizer
        self.tokenizer = tokenizer.get_tokenizer()
        self.CLS_ids = self.tokenizer.encode('[CLS]', add_special_tokens=False).ids
        self.SEP_ids = self.tokenizer.encode('[SEP]', add_special_tokens=False).ids
        self.max_len = max_len
        self.data_path = pathlib.Path(data_path)
        self.label_encoder = preprocessing.LabelEncoder()
        self.sentences, self.tags = self.process_csv()       
        self.tag_dict = self.get_tag_dict()
        

    def process_csv(self) -> (list, list):
        """Opens and processes the csv file specified by self.data_path, 
        returning a list of tokens and tags.
        
        returns
        -------
        sentences : list
            A list of lists of tokens corresponding to the sentences in the 
            input data.
        tags : list
            A list of lists of tags corresponding to the sentences.
        """
        df = pd.read_csv(self.data_path, encoding="latin-1")
        # backfill 'Sentence #' column
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
        # fit self.label_encoder
        df.loc[:, "Tag"] = self.label_encoder.fit_transform(df["Tag"])
        # get each word and tag in each sentence
        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        tags = df.groupby("Sentence #")["Tag"].apply(list).values
        return sentences, tags

    def build_train_data(self) -> (list, np.array):
        """Converts the sentences and tags produced by self.proces_csv() into 
        training inputs for the BERT model
        
        returns
        -------
        X : [np.array, np.array, np.array]
            Input of the form [token ids, token types, attention masks].
        y : np.array
            Numerically encoded tags for each sentence

        TODO: Check X_token_typ... it seems unecesary - what happens if its 
        removed? Is it definitely set up correctly?
        """
        # initialize output arrays
        X_token_ids = []
        X_token_typ = []
        X_token_msk = []
        y = []
        # build arrays for each sentence 
        for sentence, tag in zip(self.sentences, self.tags):
            ids, tags = self.encode_sentence(sentence, tag) 
            padding_len = self.max_len - len(ids)
            token_types = [0]*len(ids) + ([0]*padding_len) #see TODO
            attention_mask = [1]*len(ids) + ([0]*padding_len) # token/padding
            ids, tags = self.add_padding(ids, tags)
            # update inputs
            X_token_ids.append(np.asarray(ids))
            X_token_typ.append(np.asarray(token_types))
            X_token_msk.append(np.asarray(attention_mask))
            y.append(tags)
        # convert to numpy arrays
        X = [np.asarray(x) for x in [X_token_ids, X_token_typ, X_token_msk]]
        y = np.asarray(y)
        return X, y
    
    def build_input_from_text(self, text:str) -> list:
        """Builds a single input (X) from a given sentence. 
        
        parameters
        ----------
        text : str
            A string to encode as a BERT model input.
        
        returns
        -------
        X : [np.array, np.array, np.array]
            X input for a BERT model [token ids, token types, attention mask].
        """
        words = text.split()
        blank_tags = ['-' for word in words]
        ids, tags = self.encode_sentence(words, blank_tags, pre_tokenized=False)
        padding_len = self.max_len - len(ids)
        token_types = [0]*len(ids) + ([0]*padding_len) 
        attention_mask = [1]*len(ids) + ([0]*padding_len)      
        ids, _ = self.add_padding(ids, tags)
        X = [np.asarray(x) for x in [[ids], [token_types], [attention_mask]]]
        return X


    def encode_sentence(self, sentence:list, raw_tags:list, pre_tokenized=True) -> (list,list):
        """Converts tokens and corresponding tags into numeric inputs for the 
        BERT model.

        parameters
        ----------
        sentence : list
            A list of tokens representing individual words (or partial words) 
            within a full sentence.
        raw_tags : list
            The tags corresponding to those tokens (e.g. B-geo or I-per).
        pre-tokenized : bool
            Have the inputs already been split into wordpiece tokens (e.g. 
            ['Middle', '##s', '##borough'])?

        returns
        -------
        ids : list
            A list of numeric values corresponding to each token in a sentence.
        tags : list
            A list of numeric tags corresponding to the raw_tags for a sentence.        
        """
        encode = lambda w : self.tokenizer.encode(w, add_special_tokens=False)
        if pre_tokenized:
            # get base ids and tags
            ids = []
            for word in sentence:
                try:
                    ids.append(self.tokenizer.get_vocab()[word])
                except KeyError:
                    ids.append(self.tokenizer.get_vocab()['[unused1]'])
            ids = [self.tokenizer.get_vocab()[w] for w in sentence]
            ids = flatten(ids)
            tags = raw_tags
        else:
            ids = [encode(str(w)).ids for w in sentence]
            tags = [[t]*len(i) for t,i in zip(raw_tags, ids)]            
            # flatten into single lists
            ids = flatten(ids)
            tags = flatten(tags)
        if (sentence[0] != '[CLS]') and (sentence[1] != '[SEP]'):
            # crop to max_len -2 (allowing for CLS/SEP tags)
            ids = ids[:self.max_len - 2]        
            tags = tags[:self.max_len - 2]
            # add CLS/SEP tags and padding
            ids, tags = self.add_special_tokens(ids, tags)
        return ids, tags
    
    

    def parse_sentence(self, sentence):
        new_sentence = []
        for word in sentence:
            if str(word)[:2]=='##':
                new_sentence[-1] += word[2:]
            else:
                new_sentence.append(word)
        return new_sentence
        
        
    def add_special_tokens(self, ids:list, tags:list)->(list,list):
        """Adds [CLS] and [SEP] tokens (and corresponding tags) to start and end
        of ids/tags.

        parameters
        ----------
        ids : list
            List of ids without [CLS] and [SEP] tokens.
        tags : list
            List of tags without [CLS] and [SEP] tags. 
        
        returns
        -------
        ids : list
            List of ids with [SEP] and [CLS] tokens.
        tags : tags
            List of tags with [SEP] and [CLS] tags.
        
        """
        ids = self.CLS_ids + ids + self.SEP_ids
        tags = [self.tag_dict['O']] + tags + [self.tag_dict['O']]
        return ids, tags
    
    def add_padding(self, ids:list, tags:list)->(list,list):
        """Adds padding so that input is extended to match self.max_len
        """
        padding_tag = len(self.tag_dict)
        padding_len = self.max_len - len(ids)
        ids = ids + [0]*padding_len
        tags = tags + [padding_tag]*padding_len
        return ids, tags

    def get_tag_dict(self):
        """Returns a dictionary mapping tags (e.g. B-geo, I-per etc) to
        numeric codes used by BERT model
        
        returns
        -------
        tag_dict : dict
            Dictionary mapping tags to numeric BERT codes.
        """
        classes = self.label_encoder.classes_
        codes = self.label_encoder.transform(classes)
        tag_dict = dict(zip(classes, codes))
        return tag_dict
    
class Phrase:

    def __init__(self, token, tag):
        token = str(token)
        self.text = token.strip('##')
        if tag:
            self.tags = [tag[:2]]
        else:
            self.tags = []

    def add_token(self, token, tag):
        # add a trailing space if tag is I-Geo
        token = str(token)
        if '##' in token:
            self.update_phrase(token, tag, space=False)
        elif 'B' in tag:
            self.update_phrase(token, tag, space=False)
        else:
            self.update_phrase(token, tag, space=True)

    def update_phrase(self, token, tag, space=False):
        if space:
            self.text += ' '
        self.text += token.strip('##')
        self.tags.append(tag[:2])
        
    def clean_text(self):
        text = self.text
        if text == 'u.s.':
            text = 'United States'
        subs = ['unitary', 'authority', 'metropolitan', 
                  'micropolitan', 'area', 'province', 'of', 
                  'district', 'district', 'state']
        new_text = ''
        words = text.split(' ')
        for word in words:   
            if not word.lower() in subs:
                new_text += word + ' '
        return new_text.strip()
    
    def remove_directions(self):
        text = self.text
        subs = ['central', 'northeastern', 'southeastern', 'northwestern', 
                  'southwestern', 'eastern', 'northwest', 'southwest', 
                  'northeast', 'southeast', 'southern', 'northern', 'western', 
                  'north', 'south', 'east', 'west', 'upper', 'lower']

        new_text = ''
        words = text.split(' ')
        for word in words:   
            if not word.lower() in subs:
                new_text += word + ' '
        return new_text.strip()
        