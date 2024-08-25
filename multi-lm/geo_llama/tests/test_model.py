import unittest
import sys
import os
import json
PROJECT_PATH = os.getcwd()
sys.path.append('..')
from geo_llama.model import Model


class TestModel(unittest.TestCase):
    
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
        self.instruct_template = "This is an instruction template"
        self.input_template = r'''<text>{}<\text>
        <toponym>{}<\toponym
        <matches>{}<\matches>
        '''
        self.test_config = {'response_token':'### Response:'}
        with open('test_prompt_template.txt', 'w') as f:
            f.write(self.prompt_template)
        with open('test_instruct_template.txt', 'w') as f:
            f.write(self.instruct_template) 
        with open('test_input_template.txt', 'w') as f:
            f.write(self.input_template)
        with open('test_config.json', 'w') as f:
            json.dump(self.test_config, f)
                  
            
        self.model = Model(model_name='test_model',
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
        
    def test_model_loaded(self):
        out = self.model.model.generate()
        self.assertEqual(out[0], 'input ### Response: default output <\out>')
        
    def test_templates_loaded(self):
        self.assertEqual(self.prompt_template, self.model.prompt_template)
        self.assertEqual(self.instruct_template, self.model.instruction_template)
        self.assertEqual(self.input_template, self.model.input_template)
        
    def test_config_loaded(self):
        self.assertEqual(self.test_config, self.model.config)
        
    def test_format_prompt(self):
        output = self.model.format_prompt('a', 'b', 'c')
        expected_text = self.prompt_template.format('a', 'b', 'c')
        expected_output = {'text':[expected_text], }
        self.assertEqual(expected_output, output)
    
    def test_get_output(self):
        prompt = self.prompt_template.format('a', 'b', 'c')
        output = self.model.get_output(prompt, 'text')
        self.assertEqual(' default output ', output)
        
        
if __name__=='__main__':
    unittest.main()
