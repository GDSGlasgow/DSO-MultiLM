# standard library imports
import urllib
import requests

# TODO: Gazetteer not working, returns 403 errror

class Gazetteer:
    """Uses the Nominatim API as a gazeteer, returning location as a json upon 
    query.
    attributes:
        polygon (bool) : If true, self.query will include svg polygon coordinates 
            if available.default=False
        addressdetail (bool) : If true, self.query will return breakdown of address 
            hierarchy (city, county, state, country). default=True.
        base_url (str) : The base url being searched (Nominatim).
        
    methods:
        query(query:str)->json : Query with a location. Returns a json in the 
            nominatim location format.
    """
    def __init__(self, polygon:bool=True, addressdetails:bool=False):
        self.polygon=polygon
        self.address_details=addressdetails
        self.base_url = f'https://nominatim.openstreetmap.org/'
        self.session = requests.Session()
     
    def query(self, query:str, user_agent:str)->list:
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
        if self.polygon:
            url += '&polygon_geojson=1'
        if self.address_details:
            url += '&addressdetails=1'
        url += '&accept-language=en'
        headers={'User-agent':user_agent}
        try:
            # Adjust the timeout as needed
            r = self.session.get(url, timeout=10, headers=headers)  
            if r.status_code != 200:
                print(f"Unexpected status code: {r.status_code}")
            return r.json()
        
        except requests.RequestException as e:
            print(f"Error during Nominatim query: {e}")
            return []
