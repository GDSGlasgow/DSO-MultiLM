# standrad library imports
import json
import re
import sys
import os
from ast import literal_eval
from typing import Optional
# third party imports
# if running for testing then do not import Unsloth due to compatability issues
try:
    from unsloth import FastLanguageModel
except ImportError:
    # Use a dummy model if the library is not available
    class FastLanguageModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, text):
            return "dummy prediction"


"""Uses the GeoLlama3-7b-toponym and GeoLlama3-7b-RAG models to extract and
geolocate toponyms from English text.

pipeline:
input text -> GeoLlama3-7b-toponym -> extracted toponyms -> Nominatim lookup 
-> GeoLlama3-7b-RAG model.

For each input text, we run the toponym extraction model once, then the RAG
model one further time for each extracted toponym. Text with a large number of
toponyms may take some time to fully parse.    
"""
class Model:
    
    def __init__(self, 
                 model_name:str, 
                 prompt_path:str, 
                 instruct_path:str, 
                 input_path:str,
                 config_path:str,
                 test_mode:bool=False):
        
        self.test_mode = test_mode
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.instruct_path = instruct_path
        self.input_path = input_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        # load models and templates
        self.load_model()
        self.load_prompt_templates()
             
    def load_model(self):
        """Loads the model using UnSloths FastLanguageModel class. This should 
        work with any model hosted on huggingface. You're mileage may vary if 
        not using fine-tuned GeoLlama models.
        
        If using an OpenAI model this block could be changed to include an API
        call using an API key.
        """
        if self.test_mode:
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()
        else:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.config['max_seq_length'],
            dtype = self.config['dtype'],
            load_in_4bit = self.config['load_in_4bit'])  
            FastLanguageModel.for_inference(self.model)
        
    def load_prompt_templates(self):
        """Loads the prompts from the paths provided in __init__. Note that 
        input_path can be None as the Toponym model only uses the text as an 
        input.
        """
        with open(self.prompt_path, 'r') as f:
            self.prompt_template = f.read() 
        with open(self.instruct_path, 'r') as f:
            self.instruction_template = f.read()
        # Toponym prompt doesn't need input 
        if not self.input_path:
            return            
        with open(self.input_path, 'r') as f:
            self.input_template = f.read()  
            
    def format_prompt(self, instruction, input, response):
        """Uses the provided templaes to build a prompt to send to the 
        model. Note, response is usually empty except during fine-tuning.
        args:
            instruction (str) : geoparsing/toponym extraction procedure. 
            input (str) : usually the text to be geoparsed.
            response (str) : an example response. Usually '' unless fine-tuning
        """
        text = self.prompt_template.format(instruction, input, response)
        return { "text" : [text], }
            
    def get_output(self, prompt:str, validation_data:Optional[str]=None)->str:
        """retrieves the output from the LLM given the provided prompt. Passing
        the orginal text will also permit validation of the result, e.g. 
        checking that the extracted toponyms exist in the text.

        Args:
            prompt (str): to be passed to the model.
            text (str, optional): original text. Defaults to None.

        Returns:
            str: a json formatted string.
        """
        # pass prompt to model
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to('cuda')
        output = self.model.generate(**inputs, max_new_tokens=512, use_cache=True)
        str_output = self.tokenizer.batch_decode(output)
        # split out the response and EOS tokens
        response = str_output[0].split(self.config['response_token'])[1]
        response = response.split(self.tokenizer.eos_token)[0]
        # clean the ouput
        # TODO : change text to more generic validation_data
        if hasattr(self, 'clean_response'):
            return self.clean_response(response, validation_data)
        else:
            return response

            
class RAGModel(Model):
    """Loads and handels requests/responses from the retrieval augmented 
    generation geoparsing model. The model requires a toponym, a peice of text 
    and matches to the toponym found in a gazetteer (e.g. Nominatim).
    """
    def __init__(self, 
                 model_name:str, 
                 prompt_path:str, 
                 instruct_path:str, 
                 input_path:str, 
                 config_path:str,
                 test_mode:bool=False):
        init_args = model_name, prompt_path, instruct_path, input_path, config_path, test_mode
        # use attributes from Model object
        super().__init__(*init_args) 
        
    def geoparse_prompt(self, text:str, toponym:str, matches:list[dict])->str:
        """Generates a prompt for GeoLlama-7b-RAG using the provided text,
        toponyms and matches base don the templates specified during object
        initialisation.

        Args:
            text (str): the text to be geoparsed.
            toponym (str): the toponym from the text to be geoparsed.
            matches (list[dict]): a list of potential matches from OSM.

        Returns:
            str: the constructed geoparsing prompt.
        """
        instruction = self.instruction_template
        input = self.input_template.format(text, toponym, matches)
        return self.format_prompt(instruction, input, '')   
    
    def clean_response(self, response, validation):
        """Sanitizes an output into a dictionary with no repeated toponyms, and
        only those toponyms which appear in the text.
        args:
            response (str) : the response to be sanitized.
            validation (str) : the matches to validate against.
        returns:
            dict[str,str|float] : a formatted output.     
        """
        # fix frequently incorrect characters
        output = response.replace("'", '"').replace('True', 'true').replace('False', 'false')
        # try to read as a json, and attempt to fix otherwise.
        try:
            json_out = json.loads(output)
        except:
            json_out=  self.fix_json(output)
        
        if validation is not None: # note we don't want this to proc if []
            return self.validate_json(json_out, validation)
        else:
            return json_out
    
    def fix_json(self, json_str:str):
        """Uses the expected format of the JSON to fix potential errors, such as
        missing commas or truncated outputs
        
        args:
            json_str (str) : the json to be fixed as a string.
        return:
            dict : the fixed json as a dictionary.   
        """

        words = self.extract_words(json_str)
        # add missing words:
        expected_keys = ['name', 'latitude', 'longitude' ,'RAG_estimated']
        if any([w not in words for w in expected_keys]):
            words = self.add_missing_keys(words, expected_keys)
        ### get the values for each of the expected keys
        place_name = self.get_word(words=words, 
                                   start_word='name', 
                                   stop_word='latitude')
        latitude = self.get_word(words=words, 
                                 start_word='latitude', 
                                 stop_word='longitude')
        longitude = self.get_word(words=words, 
                                  start_word='longitude', 
                                  stop_word='RAG_estimated')
        
        bools = [w for w in words if w.lower() in ['true', 'false']] 
        
        return {"name":place_name, 
                "latitude":literal_eval(latitude), 
                "longitude":literal_eval(longitude), 
                "RAG_estimated":literal_eval(bools[0].capitalize())}  

    def add_missing_keys(self, words:list, expected_keys:list):
        """Given a list of words taken as items from a dictionary and ordered as
        [key, value, key, value,...], this function adds a [key, False] pair 
        from expected keys if it does not already appear in words. The keys will
        be added in the order they appear in expected_keys.
        Args: 
            words (list[str]) : A list of words structured as [key, value, ...].
            expected (list[str]) : a list of keys expected to be in words in order.
        Returns
            list[str] the original list with expected keys added in postion.
        """
        i = 0
        for key in expected_keys:
            if key not in words:
                # [key, False]
                words.insert(i, key)
                words.insert(i+1, 'False') 
                # increase index by two
                i+=2
            i+=2
        return words
    
    
    def extract_words(self, json_str:str)->list[str]:
        """Extracts the full words and +/- floats (inc. special characters, 
        periods, hyphens etc) from a json string. We're using quite a clunky 
        regular expression here, courtesy of gpt4. This helps with cleaning 
        broken jsons.
        args:
            json_str (str) : json to be analysed.
        returns:
            list[str] : a list of full words in the json.
        """  
        # gets just the words, including hyphens, special characters etc
        words_pattern = re.compile(r'''
        # Match words including hyphens, apostrophes, and accented characters
        (?:
            \b[\w\u00C0-\u017F]+        # Word characters
            (?:[-.\'][\w\u00C0-\u017F]+)* # Hyphens, periods, apostrophes
        )
        |
        # Match numbers, including negative and decimal numbers
        (?:
            -?                          # Optional negative sign
            \d+                         # digit(s)
            (?:\.\d+)?                  # Optional decimal point digit(s)
        )
    ''', re.VERBOSE)
        words = re.findall(words_pattern, json_str) 
        return words
    
    def get_word(self, 
                 words:list[str], 
                 start_word:Optional[str]='name', 
                 stop_word:Optional[str]='latitude')->str:
        """From a list of words, gets all the words beteen start_word and 
        end_word (non-inclusive) and joins them with a space. Used to extract
        information from a malformed dictionary string.     
        args:
            words (list) : the list of words to be analysed.
            start_word (str|None) : the word before the required word.
            stop_word (str|None) : the word after the required word.
        returns:
            str : the required word.
        """
        # initialise indices
        start_idx=-1
        stop_idx=None
        # update indices according to star/stop words 
        if start_word:
            start_idx = words.index(start_word)
        if stop_word:
            stop_idx = words.index(stop_word)
        place_words = words[start_idx+1:stop_idx]
        place_name = ' '.join([str(w) for w in place_words])
        return place_name
    
    def validate_json(self, json_dict:dict, matches:list[dict]):
        """Checks if the geographic location described in json_dict exists in 
        the list of locations in matches, to within 3 s.f. for lat/lon.
        args:
            json_dict (dict) : with 'latitude' and 'longitude' keys.
            matches (list[dict]):  with 'latitude' and 'longitude' keys.
        returns:
            dict : as json_dict, with RAG_estimated=True if location in matches.    
        """
        # if matches is an empty list set RAG as False
        if len(matches) == 0:
            json_dict.update({'RAG_estimated':False})
            return json_dict
        # otherwise check for matching coordinates
        rag=False
        for m in matches:
            d_lat = abs(float(m['lat'])-float(json_dict['latitude']))
            d_lon = abs(float(m['lon'])-float(json_dict['longitude']))
            if (d_lat <= 0.001) and (d_lon <= 0.001):
                rag =True
        json_dict.update({'RAG_estimated':rag}) 
        return json_dict
        
        
class TopoModel(Model):
    """Loads and handels requests/responses from the retrieval toponym 
    extraction model. The model requires a toponym, a peice of text and matches 
    to the toponym found in a gazetteer (e.g. Nominatim).
    """
    def __init__(self, 
                 model_name:str, 
                 prompt_path:str, 
                 instruct_path:str, 
                 input_path:str, 
                 config_path:dict,
                 test_mode:bool=False):
        init_args = model_name, prompt_path, instruct_path, input_path, config_path, test_mode
        super().__init__(*init_args)    
           
    def toponym_prompt(self, text:str)->str:
        """Builds a prompt for GeoLlama-7b-toponym using the provided text and
        the templates provided during initialization.

        Args:
            text (str): the text to be analysed.

        Returns:
            str : the formatted prompt.
        """
        instruction = self.instruction_template
        return self.format_prompt(instruction, text, '')
        
    def clean_response(self, response:str, text:str)->dict[str, list]:
        """Sanitizes an output into a dictionary with no repeated toponyms, and
        only those toponyms which appear in the text.
        args:
            response (str) : the response to be sanitized.
            text (str) : the original text.
        returns:
            dict[str,str|float] : a formatted output.     
        """
        response = response.replace("'", '"')
        try:
            output = json.loads(response)
        except:
            output = self.fix_json(response)
        # validate the toponyms against the text
        valid_toponyms = self.validate_toponyms(output['toponyms'], text)
        return {'toponyms':valid_toponyms}    
        
        
    def fix_json(self, json_str:str)->dict[str,list]:
        """Fixes a common problem in which the outputed toponym list is 
        truncated and/or contains repeated toponyms.
        
        args:
            json_str (str) : the output json string.
        returns:
            dict [str, list] : a cleaned ouput.
        """
        # get the list part of the json_str and split into list
        list_part = json_str.split('["')[1] 
        # we now want to get rid of any trailing non-alphanumerics
        for i, item in enumerate(list_part[::-1]):
            if item.isalpha() or item.isnumeric():
                break
        list_part = list_part[:-i]
        # split into list elements
        items = list_part.split('", "')
        # deduplicate
        unique_items = list(set(items))
        return {"toponyms":[t for t in unique_items if len(t) != 0]}
    
    def validate_toponyms(self, toponyms:list[str], text:str)->list[str]:
        """Removes any toponyms from the list which are not in the text.
        
        args:
            toponyms (list[str]) : a list of toponyms to check.
            text (str) : the text in which the toponyms should occur.
        reutrns:
            list[str] : a list of toponyms which occur in the text.
        """
        valid_toponyms = []
        for toponym in toponyms:
            if toponym in text:
                valid_toponyms.append(toponym)
        return valid_toponyms
        
# Dummy models for testing
class DummyModel:
    
    def __init__(self):
        self.model_type=None
    def generate(self, **kwargs):
        if self.model_type == 'toponym':
            return ["{'toponyms':['Paris', 'London', 'New York']}"]
        elif self.model_type == 'RAG':
            return ["[{'name':'Paris', 'latitude':-1.43, 'longitude':32.48},\
                    {'name':'London','latitude':-0.45, 'longitude':48.3},\
                    {'name':'New York', 'latitude':-94.34, 'longitue':'24.73'}]"]
        else:
            return ['input ### Response: default output <\out>']
        
    def get_output(self, **kwargs):
        out_str = self.generate()[0]
        out_str = out_str.replace("'", '"')
        return json.loads(out_str)
    
    def geoparse_prompt(self, **kwargs):
        return {'text':'test_text'}
    
    def toponym_prompt(self, **kwargs):
        return {'text':'test_text'}
            

class DummyTokenizer:
    
    def __init__(self):
        self.eos_token = '<\out>'
        
    def __call__(self, text, **kwargs):
        return DummyOutput()

    def batch_decode(self, output):
        return output
 
class DummyOutput:
       
    def to(self, device):
        return {}
    

    

    