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
        with patch.object(self.model, 'add_missing_keys', wraps=self.model.add_missing_keys) as mock_add_keys:
            self.model.fix_json(json_str)
            mock_add_keys.assert_called()
            
    def test_add_missing_keys_one_missing(self):
        words = ['name','Paris','longitude',2.34, 'RAG_estimated', True]
        expected_keys = ['name', 'latitude', 'longitude' ,'RAG_estimated']
        expected_out = ['name','Paris','latitude','False','longitude',2.34, 'RAG_estimated', True]
        out = self.model.add_missing_keys(words, expected_keys)
        msg1 = "Function does not add missing key"
        self.assertTrue(all([w in out for w in expected_keys]), msg1)
        msg2 = "Function does not add missing key in correct order."
        self.assertListEqual(expected_out, out, msg2)
        
    def test_add_missing_keys_multiple_missing(self):
        words = ['name','Paris','longitude',2.34]
        expected_keys = ['name', 'latitude', 'longitude' ,'RAG_estimated']
        expected_out = ['name','Paris','latitude','False','longitude',2.34, 'RAG_estimated', 'False']
        out = self.model.add_missing_keys(words, expected_keys)
        msg1 = "Function does not add missing key"
        self.assertTrue(all([w in out for w in expected_keys]), msg1)
        msg2 = "Function does not add missing key in correct order."
        self.assertListEqual(expected_out, out, msg2)       
        
    def test_extract_words_strips_characters(self):
        words = '{"name":"Paris"}'
        expected = ['name', 'Paris']
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)

    def test_extract_words_floats(self):
        words = '{"longitude":2.34}'
        expected = ['longitude', '2.34']
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)             

    def test_extract_words_negatives(self):
        words = '{"longitude": -2.34}'
        expected = ['longitude', '-2.34']
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)        
        
    def test_extract_words_hyphenated(self):
        words = '{"name": "Champs-Elysees"}'
        expected = ['name', "Champs-Elysees"]
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2) 
          
    def test_extract_words_accented(self):
        words = '{"name": "Élysées"}'
        expected = ['name', "Élysées"]
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2) 
        
    def test_extract_words_apostrophe(self):
        words = "{'name': 'd'Huez'}"
        expected = ['name', "d'Huez"]
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)     
        
    def test_extract_words_mixed_alphanumeric(self):
        words = "{'name': '129a'}"
        expected = ['name', "129a"]
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)          

    def test_extract_words_multi_word_phrase(self):
        words = "{'name': 'Alpe d'Huez'}"
        expected = ['name', "Alpe", "d'Huez"]
        out = self.model.extract_words(words)
        # check all word present
        msg1 = "Output words do not match expected."
        self.assertCountEqual(out, expected, msg1)
        # check correct order
        msg2 = "Output words not in expected order."
        self.assertListEqual(out, expected, msg2)
        
    def test_get_word_single_word(self):
        words = ['place', 'Paris', 'latitude', '44.85']
        start_word = 'place'
        stop_word = 'latitude'
        expected = 'Paris'
        out = self.model.get_word(words, start_word, stop_word)
        msg = "Expected word not returned."
        self.assertEqual(expected, out)
            
    def test_get_word_multi_word(self):
        words = ['place', 'Alpe', "d'Huez", "latitude", "44.85"]
        start_word = 'place'
        stop_word = 'latitude'
        expected = "Alpe d'Huez"
        out = self.model.get_word(words, start_word, stop_word)
        msg = "Expected word not returned."
        self.assertEqual(expected, out)
        
    def test_get_word_extreme_left(self):
        words = ['place', 'Alpe', "d'Huez", "latitude", "44.85"]
        start_word = None
        stop_word = 'latitude'
        expected = "place Alpe d'Huez"
        out = self.model.get_word(words, start_word, stop_word)
        msg = "Expected word not returned."
        self.assertEqual(expected, out) 
        
    def test_get_word_extreme_right(self):
        words = ['place', 'Alpe', "d'Huez", "latitude", "44.85"]
        start_word = 'latitude'
        stop_word = None
        expected = "44.85"
        out = self.model.get_word(words, start_word, stop_word)
        msg = "Expected word not returned."
        self.assertEqual(expected, out) 
        
    def test_validate_json_empty_list(self):
        json_dict = {'name':'paris'}
        matches = []
        expected = {'name':'paris', 'RAG_estimated':False}
        out = self.model.validate_json(json_dict, matches)
        msg = "Function doe not set RAG_estimated=True if passed empty matches."
        self.assertDictEqual(expected, out, msg)
        
    def test_validate_json_location_in_matches(self):
        json_dict = {'latitude':0, 'longitude':0}
        matches = [{'lat':40, 'lon':22}, 
                   {'lat':0.0001, 'lon':0.0001}]
        expected = {'latitude':0, 'longitude':0, 'RAG_estimated':True}
        out = self.model.validate_json(json_dict, matches)
        msg = "Function does not set RAG_estimated=True if location in matches."
        self.assertDictEqual(expected, out, msg)   
    
    def test_validate_json_location_not_matches(self):
        json_dict = {'latitude':0, 'longitude':0}
        matches = [{'lat':40, 'lon':22}, 
                   {'lat':0.01, 'lon':0.01}]
        expected = {'latitude':0, 'longitude':0, 'RAG_estimated':False}
        out = self.model.validate_json(json_dict, matches)
        msg = "Function does not set RAG_estimated=True if location not in matches"
        self.assertDictEqual(expected, out, msg) 
           
if __name__=='__main__':
    unittest.main()