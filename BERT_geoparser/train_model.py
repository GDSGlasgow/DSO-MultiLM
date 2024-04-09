"""This script brings together evrything in notebook 1 to train a new 
BERT-model. The script should be run from command line, with the dataset
used to train and test the model specified. 
"""
# standard library imports
from argparse import ArgumentParser
# thrid party imports
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import tensorflow as tf
# local imports
from BERT_geoparser.tokenizer import Tokenizer
from BERT_geoparser.data import Data
from BERT_geoparser.model import BertModel
from BERT_geoparser.analysis import Results

class Trainer:
    
    def __init__(self, data_path, model_size, cased, learning_rate, max_len, saved_model):
        """
        Args
        ----
        data_path : str 
            path to training data
        model_size : str 
            'base' or 'large' BERT model
        cased : str
            allow upper/lower casing in training data
        learning_rate : float (0 < lr < 1)
            Learning rate for model training
        max_len : int
            Maximum number of tokens for each model input. 
        saved_model : str
            Path to previously saved model (if required).
            """
        self.tokenizer = Tokenizer(model_size, cased)
        self.data = Data(data_path, self.tokenizer, max_len)
        self.model = BertModel(saved_model=saved_model, 
                               data=self.data, 
                               convolutional=True, 
                               lr=learning_rate)
    def compute_weights(self):
        """Computes class weights for data. This is useful for training on the 
        unbalanced NER datasets.
        
        args
        ----
        data : BERT_geoparser.data
            The data object used to train the model.
            
        returns
        -------
        weights : dict
            class weights for the model. 
        """
        # use sklearn to calculate class weights
        data_df = pd.read_csv(self.data.data_path, encoding='latin')
        class_weights_list = class_weight.compute_class_weight('balanced',
                                                        classes=data_df.Tag.unique(),
                                                        y=data_df.Tag.values)

        # turn into a dictionary readable by the model
        class_weights = {}
        for tag, weight in zip(data_df.Tag.unique(), class_weights_list):
            tag_numeric = self.data.tag_dict[tag]
            class_weights.update({tag_numeric:weight}) 
        # add an arbritary weight for the masked values 
        class_weights[len(self.data.tag_dict)] = 0.001
        class_weights = dict(sorted(class_weights.items()))
        return class_weights
    
    def train(self, save_as, n_epochs=5, batch_size=4, val_split=0.1):
        weights = self.compute_weights()
        self.model.train(save_as=save_as, 
                         n_epochs=n_epochs,
                         batch_size=batch_size, 
                         validation_split=val_split,
                         class_weights=weights)
        
    def test(self, test_data, results_filename, output_csv):
        """Tests the model on the data in the provided path.

        Args
        ----
        test_data : str
            Path to test data
        results_filename : str  
            Filename for results    
                
        Generates
        ---------
        Results file with summary statistics.
        """
        y_pred, y_true = self.model.test(test_data, output_csv=output_csv)
        results_df = self.model.build_output_csv(y_pred, test_data, output_csv, False)
        res = Results(results_df)
        with open(results_filename, 'w') as f:
            unique_tags =  {x for l in y_pred for x in l}
            for tag in unique_tags:
                acc_dict = res.build_accuracy_dict(tag)
                f.write(f'"{tag}" precision : {np.round(acc_dict["precision"],3)}')
                f.write('\n')
                f.write(f'"{tag}" recall : {np.round(acc_dict["recall"],3)}')
                f.write('\n')
                f.write(f'"{tag}" F1 : {np.round(acc_dict["F1"],3)}')
                f.write('\n')
                f.write('=======================')
                f.write('\n')
            f.write(f'macro average recall : {np.round(res.macro_average(), 3)}')
            f.write('\n')
            f.write(f'macro average precision : {np.round(res.micro_average_precision(),3)}')
            f.write('\n')
            f.write(f'micro average recall : {np.round(res.micro_average_recall(),3)}')
            f.write('\n')
            f.write(f'micro average precision : {np.round(res.micro_average_F1(),3)}')
    
    
    
def main(data_path, model_size, cased, learning_rate, max_len, saved_model, 
         save_as, n_epochs, batch_size, val_split, test_data, results_filename, test_csv):
    trainer = Trainer(data_path, model_size, cased, learning_rate, max_len, saved_model)
    
    trainer.train(save_as, n_epochs, batch_size, val_split)
    if test_data:
        trainer.test(test_data, results_filename, test_csv)
        
if __name__=='__main__':
    print('######################')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('######################')
    parser = ArgumentParser()
    
    parser.add_argument('-d', '--data_path', dest='data_path', type=str)
    parser.add_argument('-m', '--model_size', dest='model_size', type=str)
    parser.add_argument('-c', '--cased', dest='cased', type=bool, default=True)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float)
    parser.add_argument('-M', '--max_len', dest='max_len', type=int)
    parser.add_argument('-s', '--saved_model', dest='saved_model', default=None, type=str)
    parser.add_argument('-S', '--save_as', dest='save_as', type=str)
    parser.add_argument('-n', '--n_epochs', dest='n_epochs', type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('-v', '--val_split', dest='val_split', type=float)
    parser.add_argument('-t', '--test_data', dest='test_data', default=False, type=str)
    parser.add_argument('-r', '--results_filename', dest='results_filename', default=False, type=str)
    parser.add_argument('-o', '--output_test_csv', dest='test_csv', default=False, type=str)
    
    args = parser.parse_args()
    
    main(args.data_path, args.model_size, args.cased, args.learning_rate, 
         args.max_len, args.saved_model, args.save_as, args.n_epochs, 
         args.batch_size, args.val_split, args.test_data, args.results_filename, args.test_csv)
