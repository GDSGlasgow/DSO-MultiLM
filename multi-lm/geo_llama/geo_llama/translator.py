# third party imports
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from lingua import LanguageDetectorBuilder, Language
from tqdm import tqdm
from typing import Any, Optional

class Translator:
    
    def __init__(self, languages:list=None, model_size:str='418M', test_mode=False):
        """Detects and translates text into a required language, using the
        M2M100 model and the Lingua package. If the language is being detected
        from a pool of possible languages these can be stated to improve
        computational efficiency, otherwise leave blank to translate from any
        language. 

        Args:
            languages (list, optional): A list of potential source languages as 
            ISO-639-1 codes. Leave as None if source language is unknown.  
            Defaults to None.
            model_str (str, optional): The model being used. Can be '418M' or 
            '1.2B'. Defaults to '418M'.
        """
        if languages:
            self.languages = [getattr(Language, l.upper()) for l in languages]
        else:
            self.languages = None
        
        self.detector = self.get_detector()
        
        self.test_mode = test_mode
        if self.test_mode:
            self.model_str = 'test_model'
            self.model = DummyTranslator()
        else:
            self.model_str = f'facebook/m2m100_{model_size}'
            self.model =  M2M100ForConditionalGeneration.from_pretrained(self.model_str)
        
    def get_detector(self)-> LanguageDetectorBuilder:
        """Retrieves the language detection model. If a list of potential
        languages has been provided in the class initialisation then the 
        detector will chose from those classes.   

        Returns:
            LanguageDetectorBuilder: initialised laguage detection model.
        """
        if self.test_mode:
            return DummyLanguageDetector()
        
        if self.languages:
            detector = LanguageDetectorBuilder.from_iso_codes_639_1(*self.languages)
        else:
            detector = LanguageDetectorBuilder.from_all_languages()
            
        return detector.build()
    
    def translate(self, text:str, out_lang:str='en')->str:
        """translates text to the language defined by out_lang. Source language
        is detected automatically.  

        Args:
            text (str): text to be translated
            out_lang (str): ISO Code 639-1 of target language (e.g. "en")

        Returns:
            str: translated text in out_lang
        """
        self.src_lang = self.detect_language(text)
        if self.src_lang==out_lang:  # return text without translation
            return {'language':self.src_lang, 'translation':text}
        # build the tokenizer
        self.get_tokenizer()
        # clean up the input to split into lines and remove formatting
        lines = [l.strip('\n') for l in text.split('\n')]
        lines = [l for l in lines if len(l.strip(' '))>0]
        # loop over lines and translate
        translated_lines = []
        print(f'Translating {len(lines)} lines:')
        for line in tqdm(lines):
        
            src_tokens = self.tokenizer(line.strip('\n'), return_tensors='pt')
            lang_id = self.tokenizer.get_lang_id(out_lang)
            out_tokens = self.model.generate(**src_tokens, forced_bos_token_id=lang_id)
            out_line = self.tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
            translated_lines.append(out_line[0])
        
        out_text = '\n\n'.join(translated_lines)
        
        return {'language':self.src_lang, 'translation':out_text}
    
    def get_tokenizer(self)->M2M100Tokenizer:
        """Retrieves the tokenizer in the required source language. If the 
        
        Returns:
            M2M100Tokenizer: _description_
        """
        if self.test_mode:
            self.tokenizer = DummyTokenizer()
            return
        try:
            self.tokenizer =  M2M100Tokenizer.from_pretrained(self.model_str, 
                                                              src_lang=self.src_lang)
        except:
            print(f'Language "{self.src_lang}" not supported by M2M100 tokenizer.\
                  Attempting with default tokenizer, whcih may impact results.')
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_str)
        
    
    def detect_language(self, text:str)-> str:
        """USes the Lingua package to detect the language of the text.

        Args:
            text (str): text to be analyzed.

        Returns:
            str: iso-639-1 code of the detected language. 
        """
        lang = self.detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower()
    
### Dummy models for testing

class DummyTranslator:
    
    def generate(*args, **kwargs):
        return ['out_token_1', 'out_token_2']
    
class DummyTokenizer:
    
    def __call__(**kwargs):
        return ['out_token_1', 'out_token_2']
    
    def batch_decode(in_tokens, **kwargs):
        return in_tokens
    
    def get_lang_id(in_lang, **kwargs):
        return in_lang
    
class DummyLanguageDetector:
    
    def detect_language_of():
        return 'la'
    