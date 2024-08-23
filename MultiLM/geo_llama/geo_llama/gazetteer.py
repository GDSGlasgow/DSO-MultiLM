import urllib
import requests
from typing import Optional

"""This script contains the Gazetteer class - an object which interacts with 
either Nominatim or GeoNames to retrieve spatial information realted to a 
toponym. This class is used to build the candidate list in the RAG model.
"""

class Gazetteer:
    """Uses the either the Nominatim or GeoNames API as a gazeteer, returning 
    location as a json upon query.
    attributes:
        gazetteer_source (str) : 'nominatim' (default) or 'geonames'. Sets the 
            source of the spatial information retrieved. 
        geonames_username (Optional[str]) : required if using GeoNames as the
            gazetteer source.
        base_url (str) : The base url being searched (Nominatim or GeoNames).
        cache (dict) : initialised as an empty dict, stores previous calls to  
            prevent the model repeatedly calling the same query to the API.
            
    methods:
        query(query:str)->json : Query with a location. Returns a json in the 
            nominatim location format.
        _nominatim_query(query:str, user_agent:str)->json : uses nominatim for query.
        _geonames_query(str)->json : uses GeoNames for query.
    """
    def __init__(self, gazetteer_source:str='nominatm', 
                       geonames_username:Optional[str]=None):
        self.gazetteer_source = gazetteer_source
        # set the base URL according to gazzetteer choice
        if self.gazetteer_source == 'nominatim':
            self.base_url = f'https://nominatim.openstreetmap.org/'
        elif self.gazetteer_source== 'geonames':
            self.base_url = 'http://api.geonames.org/'
            self.username = geonames_username
            
        self.session = requests.Session()
        self.cache = {}
    
    def query(self, query:str, user_agent:Optional[str]=None)->list:
        """Sends a query to the required gazetteer API.

        Args:
            query (str): the toponym to be searched.
            user_agent (Optional[str]): required for Nominatim. This doesn't 
                need to be unique and doesn't require an account.

        Raises:
            ValueError: If GeoNames query made without self.username being set,
                or if Nominatim query made without specifying user_agent.

        Returns:
            list: The matched locations to the query toponym in the gazetteer.
        """
        # check the cahce first
        cache_out = self.cache.get(query, None)
        if cache_out:
            return cache_out
        # otherwise search in the specified gazetteer
        if self.gazetteer_source.lower() == 'nominatim':
            if not user_agent:
                raise ValueError('Please provide user_agent for Nominatim requests')
            return self._nominatim_query(query, user_agent)
        
        elif self.gazetteer_source.lower() == 'geonames':
            if not self.username:
                raise ValueError('Please provide username for GeoNames query')
            return self._geonames_query(query, self.username)
        
    def _nominatim_query(self, query:str, user_agent:str)->list:
        """Searches Nominatim for the required location, returning a json 
        formatted list. See the Nominatim documentation for more info on this.
        
        parameters:
            query (str) : phrase being searched for. 
            user_agent (str) : User identification.   
        returns:
            list[dict] : A json formatted list of all matches on Nominatim. 
        """
        formatted_query = urllib.parse.quote(query) 
        url = self.base_url + f'search?q={formatted_query}&format=json'
        url += '&accept-language=en'
        headers={'User-agent':user_agent}
        try:
            # Adjust the timeout as needed
            r = self.session.get(url, timeout=10, headers=headers)  
            if r.status_code != 200:
                print(f"Unexpected status code: {r.status_code}")
                
            self.cache.update({'query':r.json()})
            return r.json()
        
        except requests.RequestException as e:
            print(f"Error during Nominatim query: {e}")
            return []

    def _geonames_query(self, query:str)->list:
        """Searches geonames for the required location, returning a json 
        formatted list. See the Nominatim documentation for more info on this.
        
        parameters:
            query (str) : phrase being searched for.   
        returns:
            list[dict] : A json formatted list of all matches on Nominatim. 
        """
        formatted_query = urllib.parse.quote(query) 
        url = self.base_url + f'searchJSON?q={formatted_query}' 
        url += f'&username={self.username}'
        url += '&orderby=relevance'
        url += '&maxRows=20'

        try:
            # Adjust the timeout as needed
            r = self.session.get(url, timeout=10)  
            if r.status_code != 200:
                print(f"Unexpected status code: {r.status_code}")
            out = self.format_geonames_response(r.json()['geonames'])
            self.cache.update({'query':out})
            return out
        
        except requests.RequestException as e:
            print(f"Error during GeoNames query: {e}")
            return {'geonames':[]}
        
    def format_geonames_response(self, response):
        """Reformats the geonames response to be structured more like a 
        nominatim response. This helps with handling later in the pipeline
        """
        out = []
        for m in response:
            address1 = m.get('name', '')
            address2 = m.get('adminName1', '')
            address3 = m.get('countryName', '')
            address = ', '.join([address1, address2, address3])
            
            out.append({'name':m.get('name', ''),
                        'lat':m.get('lat', ''),
                        'lon':m.get('lng',''),
                        'display_name':address
                        })
        return out
    
### Dummy Gazetteer for testing

class DummyGazetteer:
    
    def __init__(self):
        pass
    
    def query(*args):
        out_1 = {'name':'name_1',
                 'lat':0,
                 'lon':0,
                 'display_name':'display_name_1'}
        
        out_2 = {'name':'name_2',
                 'lat':0,
                 'lon':0,
                 'display_name':'display_name_2'}
        
        return [out_1, out_2]
    
    
    