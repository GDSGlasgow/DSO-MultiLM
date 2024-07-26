from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from lingua import LanguageDetectorBuilder, Language

class Translator:
    
    def __init__(self, languages:list=None, model_size:str='418M'):
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
        self.model_str = f'facebook/m2m100_{model_size}'
        self.model =  M2M100ForConditionalGeneration.from_pretrained(self.model_str)
        
    def get_detector(self)-> LanguageDetectorBuilder:
        """Retrieves the language detection model. If a list of potential
        languages has been provided in the class initialisation then the 
        detector will chose from those classes.   

        Returns:
            LanguageDetectorBuilder: initialised laguage detection model.
        """
        if self.languages:
            detector = LanguageDetectorBuilder.from_iso_codes_639_1(*self.languages)
        else:
            detector = LanguageDetectorBuilder.from_all_languages()
            
        return detector.build()
    
    def translate(self, text:str, out_lang:str)->str:
        """translates text to the language defined by out_lang. Source language
        is detected automatically.  

        Args:
            text (str): text to be translated
            out_lang (str): ISO Code 639-1 of target language (e.g. "en")

        Returns:
            str: translated text in out_lang
        """
        src_lang = self.detect_language(text)
        src_tokenizer = self.get_tokenizer(src_lang)
        src_tokens = src_tokenizer(text, return_tensors='pt')
        out_tokens = self.model.generate(**src_tokens, forced_bos_token_id=src_tokenizer.get_lang_id(out_lang))
        out_text = src_tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
        
        return {'lanuage':src_lang, 'translation':out_text}
    
    def get_tokenizer(self, src_lang:str)->M2M100Tokenizer:
        """Retrieves the tokenizer in the required source language. If the 

        Args:
            src_lang (str): ISO0-639-1 country code

        Returns:
            M2M100Tokenizer: _description_
        """
        try:
            return M2M100Tokenizer.from_pretrained(self.model_str, src_lang=src_lang)
        except:
            return M2M100Tokenizer.from_pretrained(self.model_str)
        
    
    def detect_language(self, text:str)-> str:
        """USes the Lingua package to detect the language of the text.

        Args:
            text (str): text to be analyzed.

        Returns:
            str: iso-639-1 code of the detected language. 
        """
        lang = self.detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower()