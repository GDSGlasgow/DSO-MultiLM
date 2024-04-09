import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from BERT_geoparser.utils import flatten
import pandas as pd


class Results:
    
    def __init__(self, results_df:pd.DataFrame):
        self.df = results_df
        
        assert 'pred' in self.df.columns, 'Dataframe must have "pred" column.'
        assert 'tag' in self.df.columns, 'Dataframe must have "tag" column'
        
        if 'correct' not in self.df.columns:
            self.add_correct_column()

    def add_correct_column(self):
        correct = []
        for i, row in self.df.iterrows():
            if row.pred == row.tag:
                correct.append('y')
            else:
                correct.append('n')
        self.df['correct'] = correct
        return self.df

    def tag_TP(self, tag):
        """true positives for the given tag"""
        tag_predicted = self.df[self.df.pred == tag]
        TP = len(tag_predicted[tag_predicted.correct == 'y'])
        return TP
        
    def tag_FP(self, tag):
        """false positives for the given tag"""
        tag_predicted = self.df[self.df.pred == tag]
        FP = len(tag_predicted[tag_predicted.correct == 'n'])
        return FP

    def tag_FN(self, tag):
        """false negatives for the given tag"""
        tag_not_predicted = self.df[self.df.pred != tag]
        FN = len(tag_not_predicted[tag_not_predicted.tag==tag])
        return FN

    def tag_precision(self, tag):
        TP = self.tag_TP(tag)
        FP = self.tag_FP(tag)
        precision = TP/(TP+FP)
        return precision

    def tag_recall(self, tag):
        TP = self.tag_TP(tag)
        FN = self.tag_FN(tag)
        recall = TP/(TP+FN)
        return recall

    def tag_F1(self, tag):
        p = self.tag_precision(tag)
        r = self.tag_recall(tag) 
        f1 = 2*p*r/(p+r)
        return f1


    def build_accuracy_dict(self, tag):
        accuracy = {}
    
        f1 = self.tag_F1(tag)
        precision = self.tag_precision(tag)
        recall = self.tag_recall(tag)
        N_predicted = len(self.df[self.df.pred == tag])
        N_tagged = len(self.df[self.df.tag == tag])
        accuracy = {'precision':precision,
                    'recall':recall,
                    'F1':f1,
                    'total tagged':N_tagged,
                    'total predicted':N_predicted}
        return accuracy
            

    def macro_average(self, tags, measure='precision')->float:
        """Macro averaged precision across all categories.
        """
        tag_measure = []
        for tag in tags:
            tag_accuracy = self.build_accuracy_dict(tag)
            tag_measure.append(tag_accuracy[measure])
        macro_average = sum(tag_measure)/len(tags)
        return macro_average

    def micro_average_precision(self, tags)->float:
        """Micro averaged precision across all tags.
        """
        all_tag_TP = [self.tag_TP(tag) for tag in tags]
        all_tag_FP = [self.tag_FP(tag) for tag in tags]
        
        micro_average = sum(all_tag_TP)/(sum(all_tag_FP)+sum(all_tag_TP))
        return micro_average

    def micro_average_recall(self, tags)->float:
        """Micro averaged recall across all categories.
        """
        all_tag_TP = [self.tag_TP(tag) for tag in tags]
        all_tag_FN = [self.tag_FN(tag) for tag in tags]
        
        micro_average = sum(all_tag_TP)/(sum(all_tag_FN)+sum(all_tag_TP))
        return micro_average

    def micro_average_F1(self, tags)->float:
        """Micro averaged F1 across all categories.
        """
        mu_r = self.micro_average_precision(tags)
        mu_p = self.micro_average_recall(tags)
        
        micro_average = 2*mu_r*mu_p/(mu_r + mu_p)
        return micro_average


