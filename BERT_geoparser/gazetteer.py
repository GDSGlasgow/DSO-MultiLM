import urllib
import requests
import time

class Gazetteer:
    """Uses the Nominatim API as a gazeteer, returning location as a json upon 
    query.
    
    attributes
    ----------
    polygon : bool
        If true, self.query will include svg polygon coordinates if available.
        default=False
    addressdetail : bool
        If true, self.query will return breakdown of sddress hierarchy (city, 
        county, state, country). default=True
    base_url : str
        The base url being searched (Nominatim)
        
    methods
    -------
    query(query:str)->json
        Query qith a location. Returns a json in the nominatim location format.
    """
    def __init__(self, polygon:bool=True, addressdetails:bool=False):
        self.polygon=polygon
        self.address_details=addressdetails
        self.base_url = f'https://nominatim.openstreetmap.org/'
        self.session = requests.Session()
     
    def query(self, query:str)->list:
        """Searches Nominatim for the required location, returning a json 
        formatted list. See the Nominatim documentation for more info on this.
        
        parameters
        ----------
        query : str
            The phrase being searched for.    
        returns
        -------
        r.json() : list
            A json formatted list of all matches on Nominatim. 
        """
        formatted_query = urllib.parse.quote(query) 
        url = self.base_url + f'search?q={formatted_query}&format=json'
        if self.polygon:
            url += '&polygon_geojson=1'
        if self.address_details:
            url += '&addressdetails=1'
        url += '&accept-language=en'
        
        try:
            r = self.session.get(url, timeout=10)  # Adjust the timeout as needed
            if r.status_code != 200:
                print(f"Unexpected status code: {r.status_code}")
            return r
        
        except requests.RequestException as e:
            print(f"Error during Nominatim query: {e}")
            return []
        
    def reverse_lookup(self, lat, lon):
        url = self.base_url + f'reverse?format=geojson&lat={lat}&lon={lon}'
        try:
            r = self.session.get(url, timeout=10)  # Adjust the timeout as needed
            if r.status_code != 200:
                print(f"Unexpected status code: {r.status_code}")
            r.raise_for_status()  # Raise an HTTPError for bad responses
            return r
        except requests.RequestException as e:
            print(f"Error during Nominatim query: {e}")
            return []