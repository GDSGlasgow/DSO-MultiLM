import unittest
import os
import sys
import json
from unittest.mock import patch
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from geo_llama.model import RAGModel


class TestTopoModel(unittest.TestCase):
    def setUp(self):
        # add some test prompts
        self.prompt_template = '''This is a prompt template. 
        ### Instruction:
        {}
        ### Input:
        {}
        ### Response:
        {}
        '''
        self.input_template = r'''<text>{}<\text>
        <toponym>{}<\toponym
        <matches>{}<\matches>
        '''
        self.instruct_template = "This is an instruction template {} {}"
        self.test_config = {'response_token':'### Response:'}
        with open('test_prompt_template.txt', 'w') as f:
            f.write(self.prompt_template)
        with open('test_instruct_template.txt', 'w') as f:
            f.write(self.instruct_template) 
        with open('test_input_template.txt', 'w') as f:
            f.write(self.input_template)
        with open('test_config.json', 'w') as f:
            json.dump(self.test_config, f)
                  
            
        self.model = RAGModel(model_name='test_model',
                              prompt_path='test_prompt_template.txt',
                              instruct_path='test_instruct_template.txt',
                              input_path='test_input_template.txt',
                              config_path='test_config.json',
                              test_mode=True)
        
        self.model.model.model_type = 'default'
        
    def tearDown(self):
        os.remove('test_prompt_template.txt')
        os.remove('test_instruct_template.txt')
        os.remove('test_input_template.txt')
        os.remove('test_config.json')
        
    def test_geoparse_prompt(self):
        text = 'The 2024 Olympic games took place in Paris.'
        toponym = 'Paris'
        matches = [{'name':'Paris', 'latitude':48.85, 'longitude':2.34}]
        
        instruct = self.instruct_template
        input = self.input_template.format(text, toponym, matches)
        expected = {'text':[self.model.prompt_template.format(instruct, input, "")]}
        output = self.model.geoparse_prompt(text, toponym, matches)
        msg = "Toponym prompt not formed correctly."
        self.assertEqual(expected, output, msg)
        
    def test_clean_response_incorrect_quotes(self):
        json_str = "{'name':'Paris', 'latitude':48.85, 'longitude':2.34}"
        expected = {"name":"Paris", "latitude":48.85, "longitude":2.34}
        with patch.object(self.model, 'fix_json', wraps=self.model.fix_json) as mock_fix_json:
            out = self.model.clean_response(json_str, None)
            msg ="clean_response() does not sanitize quote marks."
            self.assertDictEqual(out, expected, msg)
            mock_fix_json.assert_not_called()
        
    def test_clean_response_incorrect_bool_true(self):
        json_str = "{'name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG':True}"
        expected = {"name":"Paris", "latitude":48.85, "longitude":2.34, 'RAG':True}
        with patch.object(self.model, 'fix_json', wraps=self.model.fix_json) as mock_fix_json:
            out = self.model.clean_response(json_str, None)
            # check correct fix made
            msg ="clean_response() does not sanitize 'True' to 'true'"
            self.assertDictEqual(out, expected, msg)
            # check fix_json() not called
            mock_fix_json.assert_not_called()
            
        
    def test_clean_response_incorrect_bool_false(self):
        json_str = "{'name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG':False}"
        expected = {"name":"Paris", "latitude":48.85, "longitude":2.34, 'RAG':False}
        with patch.object(self.model, 'fix_json', wraps=self.model.fix_json) as mock_fix_json:
            out = self.model.clean_response(json_str, None)
            msg ="clean_response() does not sanitize 'False' to 'false'"
            self.assertDictEqual(out, expected, msg)
            # check fix_json() not called
            mock_fix_json.assert_not_called()
        
    def test_clean_response_dict_output(self):
        json_str = "{'name':'Paris', 'latitude':48.85, 'longitude':2.34}"
        with patch.object(self.model, 'fix_json', wraps=self.model.fix_json) as mock_fix_json:
            out = self.model.clean_response(json_str, None)
            msg = "clean response does not return a dictionary."
            self.assertIsInstance(out, dict, msg)
            mock_fix_json.assert_not_called()
            
    def test_clean_response_broken_input(self):
        json_str = "{'name':'Paris', 'latitude':48.85, 'longitude':2.34', 'RAG_estimated':False"
        with patch.object(self.model, 'fix_json', wraps=self.model.fix_json) as mock_fix_json:
            out = self.model.clean_response(json_str, None)
            mock_fix_json.assert_called()
            
    def test_fix_json_missing_comma(self):
        json_str = "{'name':'Paris', 'latitude':48.85 'longitude':2.34, 'RAG_estimated':true}"
        expected = {'name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG_estimated':True}
        output = self.model.fix_json(json_str)
        self.assertDictEqual(expected, output)
        
    def test_fix_json_missing_quotes(self):
        json_str = "{name':'Paris, 'latitude':48.85 'longitude':2.34, 'RAG_estimated':true}"
        expected = {'name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG_estimated':True}
        output = self.model.fix_json(json_str)
        self.assertDictEqual(expected, output)
        
    def test_fix_json_missing_bracket(self):
        json_str = "name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG_estimated':true"
        expected = {'name':'Paris', 'latitude':48.85, 'longitude':2.34, 'RAG_estimated':True}
        output = self.model.fix_json(json_str)
        self.assertDictEqual(expected, output)    
    
    def test_fix_json_missing_word(self):
        json_str = "{'name':'Paris', 'longitude':2.34, 'RAG_estimated':true}"
        with patch.object(self.model, 'add_missing_words', wraps=self.model.add_missing_words) as mock_add_words:
            output= self.model.fix_json(json_str)
            mock_add_words.assert_called()
                 
        
      

        
if __name__=='__main__':
    unittest.main()