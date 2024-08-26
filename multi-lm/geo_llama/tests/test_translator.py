# standard library imports
import unittest
import sys
sys.path.append('..')
from unittest.mock import patch
# third party imports
from lingua import Language
# local imports
from geo_llama.translator import DummyTranslator, DummyTokenizer, Translator

class TestTranslator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_initilised_with_languages(self):
        languages = ['ENGLISH', 'CHINESE', 'FRENCH']
        translator = Translator(languages=languages, test_mode=True)
        expected = [getattr(Language, l.upper()) for l in languages]
        msg = 'Languages not initialised correctly'
        self.assertListEqual(expected, translator.languages, msg=msg)
        
    def test_initialised_wiithout_languages(self):
        translator = Translator(test_mode=True)
        msg = 'Translator does not initialise without languages.'
        self.assertFalse(translator.languages, msg=msg)
        
    def test_model_str_initialises(self):
        model_size='418M'
        translator = Translator(model_size=model_size, test_mode=True)
        expected = 'facebook/m2m100_418M'
        msg = 'Setting model_str does not initialise correctly.'
        self.assertEqual(expected, translator.model_str, msg=msg)
        
    def test_detector_initialises(self):
        translator=Translator(test_mode=True)
        msg = 'Lamguage detector not initilised correctly'
        self.assertTrue(translator.detector, msg)
        
    def test_translate_src_lang_out_lang_equal(self):
        translator=Translator(test_mode=True)
        out = translator.translate(text='test_1', out_lang='en')
        expected={'language':'en', 'translation':'test_1'}
        msg='Original text not returned when src_lang==out_lang.'
        self.assertDictEqual(expected, out, msg=msg)
        
    def test_translate_src_lang_out_lang_not_equal(self):
        translator=Translator(test_mode=True)
        out = translator.translate(text='test_1', out_lang='fr')
        expected={'language':'en', 'translation':'out_token_1 out_token_2'}
        msg='Translated text not returned when src_lang!=out_lang'
        self.assertDictEqual(expected, out, msg=msg)
        
    def test_translate_multiple_lines(self):
        translator=Translator(test_mode=True)
        out = translator.translate(text='test_1 \n\n test_2', out_lang='fr')
        expected_text = 'out_token_1 out_token_2\n\nout_token_1 out_token_2'
        expected = {'language':'en', 'translation':expected_text}
        msg='Translation of multiple lines not functioning'
        self.assertDictEqual(out, expected)
        
    def  test_translate_multiple_lines_fnc_calls(self):
        """Tests if the model generation calls are called once per line"""
        translator=Translator(test_mode=True)
        with patch.object(translator.model, 
                          'generate', 
                          wraps=translator.model.generate) as mock_generate:
            msg = translator.translate(text='test_1 \n\n test_2', out_lang='fr')
            
            self.assertEqual(mock_generate.call_count, 2, msg=msg)        
        
if __name__=='__main__':
    unittest.main()