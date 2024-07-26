<<<<<<< HEAD
#!/usr/bin/env python3
# standard library
import json
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from transformers import TFBertModel
from tqdm import tqdm
# local
from BERT_geoparser.data import Data
from BERT_geoparser.tokenizer import Tokenizer
from BERT_geoparser.utils import convert, flatten

class BertModel:
    """A class to wrap the TFBertModel object from the transformers package. It 
    is initialised with a BERT_geoparser.data.Data object, allowing for 
    training and testing of newly trianed models.

    attributes
    ----------
    loss_object : tf.keras.losses.SparseCategoricalCrossentropy
        The loss function used for training the model.
    model : tf.model
        The tensorflow model object.
    tag_dict : dict
        The dictionary mapping tags (e.g. B-geo, I-per etc) to numeric codes 
        used by the model. 
    num_tags : int
        The number of tags present in the data.
    data : BERT_geoparser.data.Data
        The data object on which to train/test the model. This is not mandatory
        if using a pre-saved model.

    """
    def __init__(self, saved_model:str=None, data:Data=None, **kwargs):
        self.loss_object = SparseCategoricalCrossentropy(from_logits=False, 
                                                         reduction=Reduction.NONE)
        if 'lr' in kwargs.keys():
            self.learning_rate=kwargs['lr']
        if 'convolutional' in kwargs.keys():
            self.convolutional=kwargs['convolutional']
        if saved_model:
            self.load_model(saved_model, data=None)
        # replace output layer if required
            if self.model.layers[-1].output_shape[-1] != self.num_tags + 1:
                predictions = layers.Dense(self.num_tags+1, activation='softmax')(self.model.layers[-2].output)
                new_model = Model(inputs=self.model.inputs, outputs=predictions)
                optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)                
                # compile
                new_model.compile(optimizer=optimizer, 
                                  loss=self.masked_ce_loss, 
                                  metrics=[self.masked_ce_loss], 
                                  weighted_metrics=[self.masked_ce_loss],)
                self.model = new_model


        # if not using pre-saved model then data object must be passed
        elif not data:
            raise ValueError("Provide path to training data if not\
                             using saved model") 
        # initialize new attributes if not using save model.
        else:
            self.data = data
            self.tag_dict = data.tag_dict
            self.num_tags = len(self.tag_dict)
            self.model = self.build_model(convolutional=self.convolutional)

    def load_model(self, saved_model, data):
        # Load a presaved model with custom ojects
        custom_objects = {"TFBertModel": TFBertModel, 
                            "masked_ce_loss":self.masked_ce_loss}

        self.model = load_model(saved_model, 
                                custom_objects=custom_objects,
                                compile=False)
        # load the model_config file
        model_name = saved_model.replace('.hdf5', '')
        with open(f'{model_name}_config.json', 'r') as f:
            config = json.load(f)        
        # initialize the data object if specified
        if data:
            self.data = data
            self.tag_dict = data.tag_dict
        # otherwise build a new Data object from the config file    
        else:
            tokenizer = Tokenizer(size=config['size'], cased=config['cased'])
            self.data = Data(tokenizer=tokenizer,
                             max_len=config['max_len'],
                             data_path=config['data_path'])
            self.tag_dict = config['tag_dict']
            
        self.num_tags = len(self.tag_dict)    
            

    def masked_ce_loss(self, true:list, pred:list)->float:
        """A wrapper around the loss object to deal with padded data.
        parameters
        ----------
        true : np.array
            True values.
        pred : np.array
            Predicted values.
        returns
        -------
        loss : float
            The loss SCCE loss between 'true' and 'pred', accounting for any 
            padding. 
        """
        # padding tag will be same as number of tags
        mask = tf.math.logical_not(tf.math.equal(true, self.num_tags))
        loss_ = self.loss_object(true, pred)
        # crop any padding
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    def build_model(self, convolutional=True)->TFBertModel:
        """Builds a BERT model which classfies indidual tokens into n_tags
        categories.

        returns
        -------
        model : tensorflow.model
            An untrained TFBertModel with n_tags categories.
        """
        ## BERT encoder
        encoder = TFBertModel.from_pretrained(str(self.data.tokenizer_obj))
        ## NER Model
        # inputs
        input_ids = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        # embedding
        embedding = encoder(input_ids, token_type_ids=token_type_ids, 
                            attention_mask=attention_mask)[0]
        embedding = layers.Dropout(0.3)(embedding)

        if convolutional:
            conv1d = layers.Conv1D(filters=16, kernel_size=3, padding='same')(embedding)
            max_pool = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d)
            #flat = layers.Flatten()(max_pool)
            embedding=layers.Dense(1024, activation='relu', name='dense_1')(max_pool)
            embedding = layers.Dropout(0.3)(embedding)

        # output
        tag_logits = layers.Dense(self.num_tags+1, activation='softmax', name='output')(embedding)
        # build model
        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[tag_logits],)
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        # compile
        model.compile(optimizer=optimizer, 
                      loss=self.masked_ce_loss, 
                      metrics=[self.masked_ce_loss], 
                      weighted_metrics=[self.masked_ce_loss],)
        return model

    def get_sample_weights(self, class_weights:dict, y:np.array):
        """Unfortunately class_weights do not work with sequential data in 
        keras (as of 02/10/2023). We can get around this by using sample weights
        and applying the appropriate class weight to each sample.
        parameters
        ----------
        class_weights : dict
            A dictionary mapping a numeric tag to a weight value.
        y : np.array
            The numeric tags for each sample in the data. 
        returns
        -------
        sample weights : list
            A list of length len(y) with the weight assigned to each sample.
        """
        sample_weights=[]
        for y_sentence in y:
            sample_weights.append([class_weights[yi] for yi in y_sentence])
        return np.asarray(sample_weights)

    
    def train(self, n_epochs:int=1, verbose:int=1, batch_size:int=16, 
              validation_split:float=0.1, save_as:str=False, class_weights=None)->None:
        """Trains the compiled TFBertModel object on the supplied data.

        parameters
        ----------
        n_epochs : int
            Number of epochs for training.
        verbose = int
            Verbosity as per Tensorflow model.
        batch_size : int
            Number of sentences per batch.
        validation_split : float < 1
            Proportion of data used in model validation.
        save_as : str
            Filename for saving model.
        
        """
        # get data
        X, y = self.data.build_train_data()
        # setup checkpoint
        checkpoint = None
        if save_as:
            checkpoint = [ModelCheckpoint(filepath=save_as,
                          save_weights_only=False,
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True)]
            self.save_config(save_as)
            
        # get sample weights
        if class_weights:
            sample_weights = self.get_sample_weights(class_weights, y)
        else:
            sample_weights = None
        # train 
        self.model.fit(X, y, 
                       epochs=n_epochs, 
                       verbose=verbose, 
                       batch_size=batch_size, 
                       validation_split=validation_split, 
                       callbacks=checkpoint,
                       sample_weight=sample_weights)
        
    def save_config(self, save_as):
        with open(f'{save_as[:-5]}_config.json', 'w') as out:
            config = {'tag_dict':self.tag_dict,
                      'size': self.data.tokenizer_obj.size,
                      'cased': self.data.tokenizer_obj.cased,
                      'data_path':self.data.data_path}
            
            json.dump(save_as, out, default=convert) 
            
    def test(self, test_data_path:str, return_tokens=False, output_csv=False)->pd.DataFrame:
        test_data = Data(test_data_path, 
                         self.data.tokenizer_obj, 
                         self.data.max_len)
        
        X_test, y_true = test_data.build_train_data()
        y_pred = self.model.predict(X_test)

        rev_tag_dict = {v:k for k,v in self.tag_dict.items()}
        y_pred_raw = []
        y_pred_tags = []
        y_true_tags = []
        X_test_tokens = []
        for i, (yi_true, yi_pred) in enumerate(zip(y_true, y_pred)):
            # strip padding from X and y:
            len_padding = len([j for j in X_test[2][i] if j==0])
            Xi_ids = X_test[0][i][:-len_padding]
            yi_true_unpadded = yi_true[:-len_padding]
            yi_pred_unpadded = yi_pred[:-len_padding]
            # update y_pred_raw
            y_pred_raw.extend(yi_pred_unpadded)
            # get the best guess for the predicted values
            yi_pred_unpadded_vals = [np.argmax(yi) for yi in yi_pred_unpadded]
            # convert from numeric labels to tags
            y_true_tags.append([rev_tag_dict[yi] for yi in yi_true_unpadded])
            y_pred_tags.append([rev_tag_dict[yi] for yi in yi_pred_unpadded_vals])
            # get the tokens form the ids and updatee list
            Xi_tokens = [self.data.tokenizer.id_to_token(x) for x in Xi_ids]
            X_test_tokens.append(Xi_tokens)
        
        if output_csv:
            out = pd.DataFrame({'token':flatten(X_test_tokens), 
                                'tag':flatten(y_true_tags), 
                                'pred':flatten(y_pred_tags)})
            for i, key in rev_tag_dict.items():
                out[key] = [y[i] for y in y_pred_raw]
            out.to_csv(output_csv, index=False)
            
        if return_tokens:
            return X_test_tokens, y_pred_tags, y_true_tags  
        else:
            return y_pred_tags, y_true_tags        

    def build_output_csv(self, y_pred, test_data_path, output_csv):
        rev_tag_dict = {v:k for k,v in self.tag_dict.items()}
        test_data = pd.read_csv(test_data_path)
        new_rows = []
        for i, y_hat_k in enumerate(y_pred):
            # get rid of [sep] token
            #y_hat_k = y_hat_k[1:]
            sentence_length = len(test_data[test_data['Sentence #']==i])
            for y_hat_m in y_hat_k[:sentence_length]:
                new_row = {}
                for j, y in enumerate(y_hat_m):
                    try:
                        new_row.update({rev_tag_dict[j]:y})
                    except KeyError:
                        new_row.update({'padding':y})
                y_hat = max(new_row, key=new_row.get)
                new_row.update({'y_hat':y_hat})
                new_rows.append(new_row)
        #new_rows_no_padding = [r for r in new_rows if r['y_hat']!='padding']
        out = pd.concat([test_data, pd.DataFrame(new_rows)], axis=1)
        print(len(new_rows))
        print(len(test_data))
        out.to_csv(output_csv)
        
            

    def predict(self, texts:list)->list:
        """Uses the model to predict the tags for a list of sentences. 
        
        parameters
        ----------
        texts : list
            A list of strings. These will be tokenized and the tags will be 
            predicted.
        
        returns
        -------
        y_pred : list
            A list of lists of tags relating to each token in the input 
            sentences.
        """
        y_pred = []
        for X_text in tqdm(texts):
            X = self.data.build_input_from_text(X_text)
            y_pred.append(self.model.predict(X, verbose=0))
        return y_pred

    def results_dataframe(self, texts:list, include_best:True)-> pd.DataFrame:
        """Builds a dataframe showing the model output across a list of textual
        inputs. Dataframe has columns "Sentence #', 'Word' and whichever tags 
        are used in the model. Each row gives the predicted probability that a 
        given word has each of the described tags.

        parameters
        ----------
        texts : list
            A list of strings to be parsed.
        include_best : bool
            If 'True' a 'Tag' column will be added with the best guess for each
            word.

        returns
        -------
        results : pd.DataFrame
            A dataframe with the probability of each tag for each word, and, if
            required, a 'Tag' column giving the best guess.

        """
        y_pred = self.predict(texts)
        results_rows = []
        
        for i_sentence, pred in enumerate(y_pred):
            X = self.data.build_input_from_text(texts[i_sentence])
            X = [xi for xi in X[0][0] if xi != 0]
            #row = {}
            for i_word, Xi in enumerate(X):
                # drop padding tags
                token = self.data.tokenizer.id_to_token(Xi)
                row = {'Sentence #':i_sentence, 'Word':token}
                for tag, i_tag in self.tag_dict.items():
                    confidence = pred[0][i_word][i_tag]
                    row.update({tag:np.round(confidence, 3)})
                
                results_rows.append(row)
        results = pd.DataFrame(results_rows)
        tags = list(self.tag_dict.keys())
        if include_best:
            best = [tags[np.argmax(r)] for r in results[tags].values]
            results['Tag'] = best
=======
#!/usr/bin/env python3
# standard library
import json
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from transformers import TFBertModel
from tqdm import tqdm
# local
from BERT_geoparser.data import Data
from BERT_geoparser.tokenizer import Tokenizer
from BERT_geoparser.utils import convert, flatten

class BertModel:
    """A class to wrap the TFBertModel object from the transformers package. It 
    is initialised with a BERT_geoparser.data.Data object, allowing for 
    training and testing of newly trianed models.

    attributes
    ----------
    loss_object : tf.keras.losses.SparseCategoricalCrossentropy
        The loss function used for training the model.
    model : tf.model
        The tensorflow model object.
    tag_dict : dict
        The dictionary mapping tags (e.g. B-geo, I-per etc) to numeric codes 
        used by the model. 
    num_tags : int
        The number of tags present in the data.
    data : BERT_geoparser.data.Data
        The data object on which to train/test the model. This is not mandatory
        if using a pre-saved model.

    """
    def __init__(self, saved_model:str=None, data:Data=None, **kwargs):
        self.loss_object = SparseCategoricalCrossentropy(from_logits=False, 
                                                         reduction=Reduction.NONE)
        if 'lr' in kwargs.keys():
            self.learning_rate=kwargs['lr']
        if 'convolutional' in kwargs.keys():
            self.convolutional=kwargs['convolutional']
        if saved_model:
            self.load_model(saved_model, data=None)
        # replace output layer if required
            if self.model.layers[-1].output_shape[-1] != self.num_tags + 1:
                predictions = layers.Dense(self.num_tags+1, activation='softmax')(self.model.layers[-2].output)
                new_model = Model(inputs=self.model.inputs, outputs=predictions)
                optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)                
                # compile
                new_model.compile(optimizer=optimizer, 
                                  loss=self.masked_ce_loss, 
                                  metrics=[self.masked_ce_loss], 
                                  weighted_metrics=[self.masked_ce_loss],)
                self.model = new_model


        # if not using pre-saved model then data object must be passed
        elif not data:
            raise ValueError("Provide path to training data if not\
                             using saved model") 
        # initialize new attributes if not using save model.
        else:
            self.data = data
            self.tag_dict = data.tag_dict
            self.num_tags = len(self.tag_dict)
            self.model = self.build_model(convolutional=self.convolutional)

    def load_model(self, saved_model, data):
        # Load a presaved model with custom ojects
        custom_objects = {"TFBertModel": TFBertModel, 
                            "masked_ce_loss":self.masked_ce_loss}

        self.model = load_model(saved_model, 
                                custom_objects=custom_objects,
                                compile=False)
        # load the model_config file
        model_name = saved_model.replace('.hdf5', '')
        with open(f'{model_name}_config.json', 'r') as f:
            config = json.load(f)        
        # initialize the data object if specified
        if data:
            self.data = data
            self.tag_dict = data.tag_dict
        # otherwise build a new Data object from the config file    
        else:
            tokenizer = Tokenizer(size=config['size'], cased=config['cased'])
            self.data = Data(tokenizer=tokenizer,
                             max_len=config['max_len'],
                             data_path=config['data_path'])
            self.tag_dict = config['tag_dict']
            
        self.num_tags = len(self.tag_dict)    
            

    def masked_ce_loss(self, true:list, pred:list)->float:
        """A wrapper around the loss object to deal with padded data.
        parameters
        ----------
        true : np.array
            True values.
        pred : np.array
            Predicted values.
        returns
        -------
        loss : float
            The loss SCCE loss between 'true' and 'pred', accounting for any 
            padding. 
        """
        # padding tag will be same as number of tags
        mask = tf.math.logical_not(tf.math.equal(true, self.num_tags))
        loss_ = self.loss_object(true, pred)
        # crop any padding
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    def build_model(self, convolutional=True)->TFBertModel:
        """Builds a BERT model which classfies indidual tokens into n_tags
        categories.

        returns
        -------
        model : tensorflow.model
            An untrained TFBertModel with n_tags categories.
        """
        ## BERT encoder
        encoder = TFBertModel.from_pretrained(str(self.data.tokenizer_obj))
        ## NER Model
        # inputs
        input_ids = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(self.data.max_len,), dtype=tf.int32)
        # embedding
        embedding = encoder(input_ids, token_type_ids=token_type_ids, 
                            attention_mask=attention_mask)[0]
        embedding = layers.Dropout(0.3)(embedding)

        if convolutional:
            conv1d = layers.Conv1D(filters=16, kernel_size=3, padding='same')(embedding)
            max_pool = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d)
            #flat = layers.Flatten()(max_pool)
            embedding=layers.Dense(1024, activation='relu', name='dense_1')(max_pool)
            embedding = layers.Dropout(0.3)(embedding)

        # output
        tag_logits = layers.Dense(self.num_tags+1, activation='softmax', name='output')(embedding)
        # build model
        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[tag_logits],)
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        # compile
        model.compile(optimizer=optimizer, 
                      loss=self.masked_ce_loss, 
                      metrics=[self.masked_ce_loss], 
                      weighted_metrics=[self.masked_ce_loss],)
        return model

    def get_sample_weights(self, class_weights:dict, y:np.array):
        """Unfortunately class_weights do not work with sequential data in 
        keras (as of 02/10/2023). We can get around this by using sample weights
        and applying the appropriate class weight to each sample.
        parameters
        ----------
        class_weights : dict
            A dictionary mapping a numeric tag to a weight value.
        y : np.array
            The numeric tags for each sample in the data. 
        returns
        -------
        sample weights : list
            A list of length len(y) with the weight assigned to each sample.
        """
        sample_weights=[]
        for y_sentence in y:
            sample_weights.append([class_weights[yi] for yi in y_sentence])
        return np.asarray(sample_weights)

    
    def train(self, n_epochs:int=1, verbose:int=1, batch_size:int=16, 
              validation_split:float=0.1, save_as:str=False, class_weights=None)->None:
        """Trains the compiled TFBertModel object on the supplied data.

        parameters
        ----------
        n_epochs : int
            Number of epochs for training.
        verbose = int
            Verbosity as per Tensorflow model.
        batch_size : int
            Number of sentences per batch.
        validation_split : float < 1
            Proportion of data used in model validation.
        save_as : str
            Filename for saving model.
        
        """
        # get data
        X, y = self.data.build_train_data()
        # setup checkpoint
        checkpoint = None
        if save_as:
            checkpoint = [ModelCheckpoint(filepath=save_as,
                          save_weights_only=False,
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True)]
            self.save_config(save_as)
            
        # get sample weights
        if class_weights:
            sample_weights = self.get_sample_weights(class_weights, y)
        else:
            sample_weights = None
        # train 
        self.model.fit(X, y, 
                       epochs=n_epochs, 
                       verbose=verbose, 
                       batch_size=batch_size, 
                       validation_split=validation_split, 
                       callbacks=checkpoint,
                       sample_weight=sample_weights)
        
    def save_config(self, save_as):
        with open(f'{save_as[:-5]}_config.json', 'w') as out:
            config = {'tag_dict':self.tag_dict,
                      'size': self.data.tokenizer_obj.size,
                      'cased': self.data.tokenizer_obj.cased,
                      'data_path':self.data.data_path}
            
            json.dump(save_as, out, default=convert) 
            
    def test(self, test_data_path:str, return_tokens=False, output_csv=False)->pd.DataFrame:
        test_data = Data(test_data_path, 
                         self.data.tokenizer_obj, 
                         self.data.max_len)
        
        X_test, y_true = test_data.build_train_data()
        y_pred = self.model.predict(X_test)

        rev_tag_dict = {v:k for k,v in self.tag_dict.items()}
        y_pred_raw = []
        y_pred_tags = []
        y_true_tags = []
        X_test_tokens = []
        for i, (yi_true, yi_pred) in enumerate(zip(y_true, y_pred)):
            # strip padding from X and y:
            len_padding = len([j for j in X_test[2][i] if j==0])
            Xi_ids = X_test[0][i][:-len_padding]
            yi_true_unpadded = yi_true[:-len_padding]
            yi_pred_unpadded = yi_pred[:-len_padding]
            # update y_pred_raw
            y_pred_raw.extend(yi_pred_unpadded)
            # get the best guess for the predicted values
            yi_pred_unpadded_vals = [np.argmax(yi) for yi in yi_pred_unpadded]
            # convert from numeric labels to tags
            y_true_tags.append([rev_tag_dict[yi] for yi in yi_true_unpadded])
            y_pred_tags.append([rev_tag_dict[yi] for yi in yi_pred_unpadded_vals])
            # get the tokens form the ids and updatee list
            Xi_tokens = [self.data.tokenizer.id_to_token(x) for x in Xi_ids]
            X_test_tokens.append(Xi_tokens)
        
        if output_csv:
            out = pd.DataFrame({'token':flatten(X_test_tokens), 
                                'tag':flatten(y_true_tags), 
                                'pred':flatten(y_pred_tags)})
            for i, key in rev_tag_dict.items():
                out[key] = [y[i] for y in y_pred_raw]
            out.to_csv(output_csv, index=False)
            
        if return_tokens:
            return X_test_tokens, y_pred_tags, y_true_tags  
        else:
            return y_pred_tags, y_true_tags        

    def build_output_csv(self, y_pred, test_data_path, output_csv):
        rev_tag_dict = {v:k for k,v in self.tag_dict.items()}
        test_data = pd.read_csv(test_data_path)
        new_rows = []
        for i, y_hat_k in enumerate(y_pred):
            # get rid of [sep] token
            #y_hat_k = y_hat_k[1:]
            sentence_length = len(test_data[test_data['Sentence #']==i])
            for y_hat_m in y_hat_k[:sentence_length]:
                new_row = {}
                for j, y in enumerate(y_hat_m):
                    try:
                        new_row.update({rev_tag_dict[j]:y})
                    except KeyError:
                        new_row.update({'padding':y})
                y_hat = max(new_row, key=new_row.get)
                new_row.update({'y_hat':y_hat})
                new_rows.append(new_row)
        #new_rows_no_padding = [r for r in new_rows if r['y_hat']!='padding']
        out = pd.concat([test_data, pd.DataFrame(new_rows)], axis=1)
        print(len(new_rows))
        print(len(test_data))
        out.to_csv(output_csv)
        
            

    def predict(self, texts:list)->list:
        """Uses the model to predict the tags for a list of sentences. 
        
        parameters
        ----------
        texts : list
            A list of strings. These will be tokenized and the tags will be 
            predicted.
        
        returns
        -------
        y_pred : list
            A list of lists of tags relating to each token in the input 
            sentences.
        """
        y_pred = []
        for X_text in tqdm(texts):
            X = self.data.build_input_from_text(X_text)
            y_pred.append(self.model.predict(X, verbose=0))
        return y_pred

    def results_dataframe(self, texts:list, include_best:True)-> pd.DataFrame:
        """Builds a dataframe showing the model output across a list of textual
        inputs. Dataframe has columns "Sentence #', 'Word' and whichever tags 
        are used in the model. Each row gives the predicted probability that a 
        given word has each of the described tags.

        parameters
        ----------
        texts : list
            A list of strings to be parsed.
        include_best : bool
            If 'True' a 'Tag' column will be added with the best guess for each
            word.

        returns
        -------
        results : pd.DataFrame
            A dataframe with the probability of each tag for each word, and, if
            required, a 'Tag' column giving the best guess.

        """
        y_pred = self.predict(texts)
        results_rows = []
        
        for i_sentence, pred in enumerate(y_pred):
            X = self.data.build_input_from_text(texts[i_sentence])
            X = [xi for xi in X[0][0] if xi != 0]
            #row = {}
            for i_word, Xi in enumerate(X):
                # drop padding tags
                token = self.data.tokenizer.id_to_token(Xi)
                row = {'Sentence #':i_sentence, 'Word':token}
                for tag, i_tag in self.tag_dict.items():
                    confidence = pred[0][i_word][i_tag]
                    row.update({tag:np.round(confidence, 3)})
                
                results_rows.append(row)
        results = pd.DataFrame(results_rows)
        tags = list(self.tag_dict.keys())
        if include_best:
            best = [tags[np.argmax(r)] for r in results[tags].values]
            results['Tag'] = best
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
        return results