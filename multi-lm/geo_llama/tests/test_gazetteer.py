# standard library
import requests
import sys
import unittest
from unittest.mock import patch
sys.path.append('..')
# third party imports
import requests_mock
# local imports
from geo_llama.gazetteer import Gazetteer

# set up a mock request function
@requests_mock.mock()
def mock_request(m, url):
    m.get(url, text='success')
    return requests.get(url).text

class TestGazetteer(unittest.TestCase):
    
    def setUp(self):
        self.nom_gaz = Gazetteer(gazetteer_source='nominatim')
        self.gnm_gaz = Gazetteer(gazetteer_source='geonames', 
                                 geonames_username='test_name')
    
    def tearDown(self):
        pass
    
    def test_initialises(self):
        """Tests if the the correct gazetteer is initialised when a given 
        gazetteer_source is passed
        """
        # nominatim
        base_url = self.nom_gaz.base_url
        expected = 'https://nominatim.openstreetmap.org/'
        self.assertEqual(base_url, expected)
        
        # geonames
        base_url = self.gnm_gaz.base_url
        expected = 'http://api.geonames.org/'
        self.assertEqual(base_url, expected)
        
        # initialized with empty cache
        self.assertDictEqual(self.gnm_gaz.cache, {})
        self.assertDictEqual(self.nom_gaz.cache, {})
    
    def test_cached_query(self):
        """Tests if the cache is used when required"""
        self.nom_gaz.cache['test_key'] = 'test_value'
        out =self.nom_gaz.query('test_key')
        expected = 'test_value'
        self.assertEqual(out, expected)
        # empty cache
        self.nom_gaz.cache = {}
        
    def test_nominatim_user_agent_missing(self):
        """Tests if missing the user_agent on nominatim queries raises an error
        """
        # function to reproduce error
        def produce_error():
            self.nom_gaz.query('test')
        self.assertRaises(ValueError, produce_error)
        
    def test_geonames_username_missing(self):
        """Tests if missing username causes geonames gazetteer to raise error"""
        gaz = Gazetteer(gazetteer_source='geonames')
        def produce_error():
            gaz.query('test')
        self.assertRaises(ValueError, produce_error)
        
    def test_build_url_calls_correcly(self):
        ### nominatim
        with patch.object(self.nom_gaz, 
                          '_build_nominatim_url', 
                          wraps=self.nom_gaz._build_nominatim_url) as mock_nom_url:
            self.nom_gaz.build_url('test')
            mock_nom_url.assert_called()
            
        ### geonames
        self.gnm_gaz = Gazetteer(gazetteer_source='geonames')
        with patch.object(self.gnm_gaz, 
                          '_build_geonames_url', 
                          wraps=self.gnm_gaz._build_geonames_url) as mock_gnm_url:
            self.gnm_gaz.build_url('test')
            mock_gnm_url.assert_called()
            
    def test_build_url_builds_correctly(self):
        ### nominatim
        nom_url = self.nom_gaz.build_url(query='test_query')
        expected = 'https://nominatim.openstreetmap.org/'
        expected += 'search?q=test_query&format=json&accept-language=en'
        self.assertEqual(nom_url, expected)
        ### geonames
        gnm_url = self.gnm_gaz.build_url(query='test_query')
        expected = 'http://api.geonames.org/'
        expected += 'searchJSON?q=test_query'
        expected += '&username=test_name&orderby=relevance&maxRows=20'
        self.assertEqual(gnm_url, expected)
        
    @requests_mock.mock()  
    def test_nominatim_query_calls(self, m):
        """Tests if the query() calls the correct gazetteer api"""
        url = self.nom_gaz.build_url('test')
        json_out = '[{"name":"test"}]'
        m.get(url, text=json_out)
        with patch.object(self.nom_gaz, 
                          '_nominatim_query', 
                          wraps=self.nom_gaz._nominatim_query) as mock_nom_query:
            _ = self.nom_gaz.query('test', user_agent='gl-test')
            mock_nom_query.assert_called()
            
    @requests_mock.mock()  
    def test_nominatim_bad_query(self, m):
        """Tests if bad _nominatim_query() call returns empty list"""
        url = self.nom_gaz.build_url('test')
        # badly constructed json will force exception
        json_out = 'bad_output'
        m.get(url, text=json_out)
        # check output is empty list
        out = self.nom_gaz.query('test', user_agent='test')
        self.assertTrue(len(out)==0, msg='Bad request does not return empty list')
            
    @requests_mock.mock()  
    def test_geonames_query_calls(self, m):
        """Tests if the query() calls the correct gazetteer api"""
        url = self.gnm_gaz.build_url('test')
        json_out = '{"geonames":[{"name":"test"}]}'
        m.get(url, text=json_out)
        with patch.object(self.gnm_gaz, 
                          '_geonames_query', 
                          wraps=self.gnm_gaz._geonames_query) as mock_gnm_query:
            _ = self.gnm_gaz.query('test')
            mock_gnm_query.assert_called()
            
    @requests_mock.mock()  
    def test_geonames_bad_query(self, m):
        """Tests if bad _geonames_query() call returns empty list"""
        url = self.gnm_gaz.build_url('test')
        # badly constructed json will force exception
        json_out = 'bad_output'
        m.get(url, text=json_out)
        # check output is empty list
        out = self.gnm_gaz.query('test')
        self.assertTrue(len(out)==0, msg='Bad request does not return empty list')                     
        
    def test_format_geonames_response(self):
        """ Tests that geonames responses are formatted correctly."""

        input_response_1 = {'name':'name_1',
                          'adminName1':'state_1',
                          'countryName':'country_1',
                          'lat':0,
                          'lng':0}
        input_response_2 = {'name':'name_2',
                          'countryName':'country_2',
                          'lat':1,
                          'lng':1}
        input_response = [input_response_1, input_response_2]
        
        output = self.gnm_gaz.format_geonames_response(input_response)
        
        expected_1 = {'name':'name_1',
                      'lat':0,
                      'lon':0,
                      'display_name':'name_1, state_1, country_1'}
        expected_2 = {'name':'name_2',
                      'lat':1,
                      'lon':1,
                      'display_name':'name_2, country_2'}
        
        msg1 = 'Incorrect number of items returned.'
        msg2 = 'Input with complete information incorrectly formatted'
        msg3 = 'Input with incomplete information incorrectly formatted'
        self.assertTrue(len(output)==2, msg=msg1)
        self.assertDictEqual(expected_1, output[0], msg=msg2)
        self.assertDictEqual(expected_2, output[1], msg=msg3)
    
    
        
if __name__=='__main__':
    unittest.main()
        