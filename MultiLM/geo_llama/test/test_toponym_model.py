import unittest
import os
import sys
import json
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from geo_llama.model import TopoModel

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
        self.input_template = 'this is an input template'
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
                  
            
        self.model = TopoModel(model_name='test_model',
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
        
    def test_toponym_prompt(self):
        text = 'a'
        instruct = self.instruct_template
        expected = {'text':[self.model.prompt_template.format(instruct, text, '')]}
        output = self.model.toponym_prompt(text)
        msg = "Toponym prompt not formed correctly."
        self.assertEqual(expected, output, msg)
    
    def test_clean_response_incorrect_quotes(self):
        json_str = "{'toponyms':['b','c','d']}"
        out = self.model.clean_response(json_str, 'bcd')
        expected = {"toponyms":["b", "c", "d"]}
        msg ="clean_response() does not sanitize quote marks."
        self.assertDictEqual(out, expected, msg)
        
    def test_clean_response_dict_output(self):
        json_str = "{'toponyms':['b','c','d']}"
        out = self.model.clean_response(json_str, 'bcd')
        msg = "clean response does not return a dictionary."
        self.assertIsInstance(out, dict, msg)
        
    def test_fix_json_truncated_input(self):
        json_str = '{"toponyms":["toponym1", "toponym2", "top]}'
        expected_keys = ['toponyms']
        expected_values = ['toponym1', 'toponym2', 'top']
        output = self.model.fix_json(json_str)
        output_keys = list(output.keys())
        output_values = list(output.values())[0]
        msg1 = "fix_json() does not return expected dictionary keys when input is truncated."
        msg2 = "fix_json() does not return expected dictionary values when input is truncated."
        self.assertCountEqual(output_keys, expected_keys, msg1)
        self.assertCountEqual(output_values, expected_values, msg2)
    
    def test_fix_json_repeated_toponyms(self):
        json_str = '{"toponyms":["toponym1", "toponym2", "toponym1"]}'
        expected_keys = ['toponyms']
        expected_values = ['toponym1', 'toponym2']
        output = self.model.fix_json(json_str)
        output_keys = list(output.keys())
        output_values = list(output.values())[0]
        msg1 = "fix_json() does not de-duplicate toponyms."
        msg2 = "fix_json() does not de-duplicate toponyms."
        self.assertCountEqual(output_keys, expected_keys, msg1)
        self.assertCountEqual(output_values, expected_values, msg2)       
    
    def test_validate_toponyms_with_invalid_toponyms(self):
        toponyms = ['London', 'Rio de Janeiro', 'Tokyo', 'Paris']
        text = 'London (2012), Rio de Janeiro (2016), Tokyo (2020)'
        output = self.model.validate_toponyms(toponyms, text)
        expected = ['London', 'Rio de Janeiro', 'Tokyo']
        msg = "Toponyms not present in text have not been removed."
        self.assertCountEqual(output, expected, msg)
        
    def test_validate_toponyms_with_valid_toponyms(self):
        toponyms = ['London', 'Rio de Janeiro', 'Tokyo', 'Paris']
        text = 'London (2012), Rio de Janeiro (2016), Tokyo (2020), Paris (2024)'
        output = self.model.validate_toponyms(toponyms, text)
        expected = ['London', 'Rio de Janeiro', 'Tokyo', 'Paris']
        msg = "Toponyms present in text have been removed."
        self.assertCountEqual(output, expected, msg)
        
    
if __name__ =='__main__':
    unittest.main()