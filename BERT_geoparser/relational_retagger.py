<<<<<<< HEAD
# standard library imports
from copy import copy
import json
# third party imports
import pandas as pd
import numpy as np
from shapely.geometry import shape, box, Point, Polygon, MultiPolygon
from tqdm import tqdm
from geopy.distance import distance
# local imports
from BERT_geoparser.data import Phrase
from BERT_geoparser.gazetteer import Gazetteer
from functools import partial
from shapely import ops
import pyproj

def reproject(geom, from_proj=None, to_proj=None):
    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)


class Location:
    """A class to handel a location object. A given location (shapely polygon)
    can be compared against othe rlocations (polygons) to identify the spatial
    relationship between the two.
    
    parameters
    ----------
    shape : shapely.geometry.(Multi)Polygon | (Multi)Point | (Multi)linestring
        A shapely geometry object descirbing the spatial characteristics of a 
        locaiton.
    """
    def __init__(self, shape):        
        self.polygon = shape
        #print(self.polygon.geom_type)
        
    def is_target(self, test_location):
        """Tests to see if the test_location is the same location as the object,
        defined as intersection comprising >80% of both polygons.
        
        parameters
        ----------
        test_location : Location
        
        returns
        -------
        bool : True if the test location matches the target
        """

        test_area = test_location.polygon.area
        if test_location.polygon.area == 0:
            return False
        true_area = self.polygon.area
        intersection = test_location.polygon.intersection(self.polygon)
        
        test_coverage = intersection.area/test_area
        true_coverage = intersection.area/true_area
        return (test_coverage > 0.8) and (true_coverage > 0.8)
    
    def is_parent(self, test_location):
        """Tests whether test_location is a parent location to self.polygon
        (i.e. self.polygon is contained within test_location.polygon)
        
        parameters
        ----------
        test_location : Location
            The Location object to be tested against.
        returns
        -------
        Bool : True if test_location is parent of self.
        """
        test_poly = test_location.polygon
        
        if 'Polygon' not in test_poly.geom_type:
            return False
        if 'Polygon' not in self.polygon.geom_type:
            return False
        # test for full containment
        if self.contains_full_area(test_poly):
            return True
        # test for centroid containment (polygon contains centroid and >50% of area)
        if self.contains_centroid(test_poly)\
           and self.contains_half_area(test_poly):
            return True
        # otherwise not a child - return false
        return False
    
    def is_child(self, test_location):
        """Tests whether test_location is a child location of the object 
        location (i.e. self.polygon contains test_location.polygon)

        parameters
        ----------
        test_location : Location
            The location to be tested against.

        Raises
        ------
        AttributeError:'A Point object can not be a parent.
            If self.polygon is type Point

        Returns
        -------
        Bool : True if test_location is a child of self.polygon
        """
        location_copy = copy(self)
        
        return test_location.is_parent(location_copy)
    
    def contains_centroid(self, poly):
        """Tests if poly contains the location polygon centroid."""
        return poly.contains(self.polygon.centroid)
    
    def contains_full_area(self, poly):
        """Tests if the test polygon contians the full area of self.polygon."""
        # first need to fill any interior holes in the polygons
        if self.polygon.geom_type == 'MultiPolygon':
            mp1 = [Polygon(p) for p in list(self.polygon.geoms)]
            p1 = MultiPolygon([Polygon(p.exterior) for p in mp1])
        elif self.polygon.geom_type == 'Polygon':
            p1 = Polygon(self.polygon.exterior)
        else:
            p1 = self.polygon
        # and do the test poly
        if poly.geom_type == 'MultiPolygon':
            mp2= [Polygon(p) for p in list(poly.geoms)]
            p2 = MultiPolygon([Polygon(p.exterior) for p in mp2])
        elif poly.geom_type == 'Polygon':
            p2 = Polygon(poly.exterior)
        else:
            # a point or linestring can not be a parent
            return False
        # return true if p2 contains p1
        return p2.contains(p1)
    
    def contains_half_area(self, poly):
        """Tests if poly contains half the area of the location polygon."""
        intersection = poly.intersection(self.polygon)
        return intersection.area/self.polygon.area >= 0.5
        
    
    def is_adjacent(self, test_location):
        """Tests whether there the test locaiton is adjacent to the current 
        location; defined as either sharing an edge, or having some interesection
        with less than half"""
        # not adjacent if it is a parent or child location
        test_poly = test_location.polygon
        if test_poly.geom_type == 'Point':
            return False
        if self.is_child(test_location):
            return False
        if self.is_parent(test_location):
            return False
        # adjacent if there is a shared boundary
        if self.has_shared_boundary(test_poly):
            return True
        # also adjacent if there is an intersection without parent/child relation
        if test_poly.intersects(self.polygon):
            return True
        return False
    
    def min_distance(self, test_poly):
        polygon1_m = reproject(self.polygon, 'EPSG:4326', 'EPSG:26944')
        polygon2_m = reproject(test_poly, 'EPSG:4326', 'EPSG:26944')
        distance_in_meters = polygon1_m.distance(polygon2_m)
        return distance_in_meters
    
    def has_shared_boundary(self, poly):
        """tests if the test location has a shared boundary with the location"""
        return poly.intersection(self.polygon).geom_type == 'LineString'
    
    def is_crossing(self, test_linestring):
        return test_linestring.polygon.crosses(self.polygon)
    
    def get_relationship(self, test_location):
        if self.is_target(test_location):
            return 'TAR'
        if self.is_parent(test_location):
            return 'PAR' # note tagging is reversed - test_location is child
        if self.is_child(test_location):
            return 'CHI'
        if (self.is_adjacent(test_location)) and not (self.is_crossing(test_location)):
            return 'ADJ'
        if self.is_crossing(test_location):
            return 'CRO'
        # if no relationship return incidental
        return 'INC'

class Retagger:

    def __init__(self, tagged_df:pd.DataFrame, reference_df:pd.DataFrame):
        self.df = tagged_df
        self.reference_df = reference_df
        self.gazetteer = Gazetteer(polygon=True, addressdetails=False)
        try:
            self.search_history = self.load_search_history()
            print("Previous search history succesfully loaded")
        except:
            self.search_history = dict()

    def location_only_results(self, location_tags)->pd.DataFrame:
        """Returns only the results which are tagged as locations]

        parameters
        ----------
        location_tags : list(str)
            Tags used to identify locations in results dataframe.
        returns
        -------
        location_results : pd.DataFrame
            Results dataframe only including rows tagged with the location tags.
        """
        tags_regex = ''.join(f'{x}|' for x in location_tags)[:-1]
        location_results = self.df[self.df.Tag.str.contains(tags_regex, regex=True)]
        return location_results
    
    def get_true_location(self, sentence_number):
        """Uses the refence dataframe to get the name and coordinates of the 
        'true' location associated with the current sentence (i.e. the name
        of the wikipedia page the sentence was taken from).

        parameters
        ----------
        sentence_number : int
            The value in column 'Sentence #' in the results daatframe. This will
            refer to a specific row in the review_df dataframe.
        returns
        -------
        dict : {name:str, coords:(float,float))
            The name of the city and the (long, lat) coordinates.
        """
        reference = self.reference_df.loc[sentence_number]
        coords = reference.coordinates
        if isinstance(coords, str):
            coords = eval(coords)
        name = reference.city
        return {'name':name,'coords':coords}

    def get_true_polygon(self, true_name:str, true_coordinates:tuple)->list:   
        """Returns a polygon from Nominatim associated with the given name and
        containing the given coordinates. Note that if Nominatim only contains a 
        Point object for that location then the bounding box is returned.
        
        parameters
        ----------
        true_name : str
            The name of the location as given in the reference dataset.
        true_coordinates : tuple (float, float)
            The lat/lon coordinates of the location given in the refence data.
            
        returns
        -------
        polygon_matches : list [shpaely.geometry.Polygon/MultiPolygon]
            All polygons found matching the given name which contain the given 
            coordinate. 
        """
        query_data = self.gazetteer.query(true_name)
        polygon_matches = []
        for match in query_data.json():
            polygon = shape(match['geojson'])
            # if the geometry is point then use Bounding Box
            if polygon.geom_type=='Point':
                miny, maxy, minx, maxx = match['boundingbox']
                polygon = box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
                
            if polygon.contains(Point(true_coordinates[::-1])):
                polygon_matches.append(polygon)
                
        if len(polygon_matches)==0:
            return('Failed')
        elif len(polygon_matches)==1:
            return polygon_matches[0]
        else:
            return self.disambiguate_true_polygons(polygon_matches)

    
    def disambiguate_true_polygons(self, polygon_matches):
        """The basic way of disambiguating is to assume the more specific area
        is the true location. We'll do that by comparing the area covered by the 
        polygons"""
        areas = [p.area for p in polygon_matches]
        return polygon_matches[np.argmin(areas)]
        
            
    def retag(self, location_tags:list, offset:int, limit:int, save_period:int=2000):
        """This function builds phrases out of the tokens tagged with location
        tags in the results dataframe and uses nominatim to find the location
        reffered to by the word or phrase. It then builds a set of set of tags 
        depending on the spatial relationship between th etarget and the 
        location. 

        parameters
        ----------
        location_tags : list
            A list of the tags used in the results dataframe to idtentify a 
            location, or location-like, word. e.g. ['geo', 'gpe, 'org']. This
            does not need to include the 'B-' or 'I-' part of the tag. 
        offset : int
            If you need to start from part way through the dataset (for example 
            if a previous run failed due to connection etc) then this param will
            allow you to adjust the starting point for the retagging.
        limit : int
            Similar to offset, this will allow you to stop the process early if 
            required.
        save_period : int
            Allows you to set a period over which the search results will be
            saved. This is a large JSON file, so a short period will seriously
            slow down the process.

        updates
        -------
        self.results : pd.DataFrame
            Updates the 'Tag' column to provide relational tags. The old tags 
            (usually B-LOC etc) are now in the column 'old_tags'. Relational
            tags go in the 'Tag' column. 
        """
        # cut down to results tagged as locations as new Retagger object
        loc_only_results = Retagger(self.location_only_results(location_tags),
                                    self.reference_df)
        # add a column identifying sequential groups (phrases) in the dataframe
        loc_only_results.add_sequential_groups()
        loc_df = loc_only_results.df

        # loop over the phrases (sequential groups) in the location only data.
        new_tags = []
        tags_idx = []
        groups = loc_df.sequential_group.unique()
        j=0
        for group in tqdm(groups[offset:limit]):
            # get just the rows in that phrase/group
            phrase_df = loc_df[loc_df.sequential_group==group].copy()
            # get the review/sentence number that phrase belongs to
            sentence_num = phrase_df['Sentence #'].iloc[0]
            
            true_loc = self.get_true_location(sentence_num)
            # initialise new Phrase object and build phrase
            phrase = Phrase(token='', tag=None)
            for word, tag in phrase_df[['Word', 'Tag']].values:
                phrase.add_token(token=word, tag=tag)

            # first check for a target phrase (phrase == true_loc['name'])
            self_string = remove_parentheses(true_loc['name'].lower())
            test_string = remove_parentheses(phrase.text.lower())
            if self_string == test_string:
                new_tags += [prefix + 'TAR' for prefix in phrase.tags]
            # build the relational tags associated with the phrase
            else:
                new_tags += self.build_tags(phrase, true_loc)
                
            tags_idx += list(phrase_df.index.values)
            j += 1
            # save the search history every 100 iterations
            if save_period:
                if j%save_period == 0:
                    self.save_search_history()
            
        # update self.df with the new tags
        self.df['new_tag'] = 'O'
        self.df.loc[tags_idx, 'new_tag'] = new_tags
        if 'old_tag' in self.df.columns:
            self.df.drop('old_tag', axis=1, inplace=True)
        self.df.rename(mapper={'Tag':'old_tag', 'new_tag':'Tag'},
                       axis='columns', 
                       inplace=True)

    
    def add_sequential_groups(self):
        """Adds a new column to self.df which identifies seperate phrases in the
        text. It assumes that a single phrase will be a collection of tokens
        with sequential indexes. As such, data with indices [0,1,2,5,6,8,10,11]
        will be given sequential_group values [0,0,0,1,1,2,3,3]. This is desinged
        to work with data that has already been cut down to location only 
        tokens using self.location_only_results.
        
        updates
        -------
        self.df : pd.DataFrame
            Adds new column 'sequential_group' indicating the phrase each token
            belongs to.
        """
        groups = []
        current_group = 0
        for index in self.df.index:
            if len(groups) == 0:
                groups.append(current_group)
            elif index == self.df.index[self.df.index.get_loc(index) - 1] + 1:
                groups.append(current_group)
            else:
                current_group += 1
                groups.append(current_group)
        
        self.df['sequential_group'] = groups


    def build_tags(self, phrase, true_loc):
        """Builds the ['B-tar', 'B-inc', 'I-tar', 'I-inc'] tags for the data, based
        on proximity to the true location provided. This uses Nominatim to get all 
        the matches for a phrase, if any pass the proximity check then a 'tar' tag
        is given, otherwise a 'inc' tag is given.
        
        parameters
        ----------
        phrase : Phrase
            A Phrase object to build tags for.
        true_loc : dict {'name':str, 'coords':tuple}
            Name and location of place referenced by the text. 
              
        returns
        -------
        list : a list of tags for the provided phrase.
        """
        # get the true polygon associated the the full sentence
        
        true_poly = self.get_true_polygon(true_loc['name'], true_loc['coords'])
        # assume INC if location not found
        if true_poly == 'Failed':
            return [prefix + 'INC' for prefix in phrase.tags]
        true_location = Location(true_poly)
        # first check if phrase.text is the same string as the target:
        if phrase.text.lower() == true_loc['name'].lower():
            return [prefix + 'TAR' for prefix in phrase.tags]
        # check for city/county child relationships
        elif true_loc['name'].lower() == str(phrase.text + ' county').lower():
            return [prefix + 'CHI' for prefix in phrase.tags]
        # find all locations matching the phrase
        if phrase.text not in self.search_history.keys():
            try:
                query_data = self.search_gazetteer(phrase)
            except:
                print(f'Unable to search query "{phrase.text}"')
                return ['MISSING' for tag in phrase.tags]
        else:
            # otherwise use the stored data
            query_data = self.search_history[phrase.text]
        relations = []
        
        # get the relationship between each match and the true location
        for match in query_data:
            if isinstance(match, str):
                print(f'Unable to search query "{phrase.text}"')
                return ['MISSING' for tag in phrase.tags]
            try:
                test_poly = shape(match['geojson'])
            except KeyError:
                relations.append('NON')
                continue
            test_location = Location(test_poly)
            relations.append(true_location.get_relationship(test_location))
        non_inc_relations = set([x for x in relations if x != 'INC'])
        if set(relations) == {'INC'}:
            tag = 'INC'
        # if there is only one none INC relation, set as that.
        elif len(non_inc_relations)==1:
            tag = list(non_inc_relations)[0]

        # for conflicting matches, choose TAR then PAR then ADJ then CHI
        elif 'TAR' in relations:
            tag = 'TAR'
        elif 'PAR' in relations:
            tag = 'PAR'
        elif 'CRO' in relations:
            tag = 'CRO'
        elif 'ADJ' in relations:
            tag = 'ADJ'
        elif 'CHI' in relations:
            tag = 'CHI'
        else:
            tag = 'NON'
        return [prefix + tag for prefix in phrase.tags]
    
    def search_gazetteer(self, phrase):
        """Searches the OSM Nominatim API for a phrase, and adjusts the phrase
        if locations can not be found.
        
        parameters
        ----------
        phrase : Phrase object
            A data.Phrase object containg the search term.
            
        returns
        -------
        query_data : [dict]
            The JSON outputted by Nominatim given the search term. If no the 
            phrase is not matched to a location then an empty list is returned.
        """
        query_data = self.gazetteer.query(phrase.text).json()
        if not isinstance(query_data, list):
            query_data = [query_data]
        # also search for locations without common suffixes and prefixes
        # if no places are found, try removing unnecesary words
        if phrase.text != phrase.clean_text():
            query_data.extend(self.gazetteer.query(phrase.clean_text()).json())
        if phrase.text != phrase.remove_directions():
            query_data.extend(self.gazetteer.query(phrase.remove_directions()).json())
        if '-' in phrase.text:
            split_phrase = phrase.clean_text().split('-')
            for p in split_phrase:
                query_data.extend(self.gazetteer.query(p).json())
        # add the search results to the search history
        self.search_history[phrase.text] = query_data
        return query_data
    
    def save_search_history(self):
        with open('search_history.json', 'w') as j:
            json.dump(self.search_history, j)
            
    def load_search_history(self):
        with open('search_history.json', 'r') as j:
            search_history = json.load(j)       
        return search_history
            

    
    
def remove_parentheses(input_string):
    """Removes everything inside parenthesese from a string"""
    result = ""
    stack = []
    for char in input_string:
        if char == '(':
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()
            else:
                result += char
        elif not stack:
            result += char
    
=======
# standard library imports
from copy import copy
import json
from functools import partial
# third party imports
import pandas as pd
import numpy as np
from shapely.geometry import shape, box, Point, Polygon, MultiPolygon
from tqdm import tqdm
from geopy.distance import distance
from shapely import ops
import pyproj
# local imports
from BERT_geoparser.data import Phrase
from BERT_geoparser.gazetteer import Gazetteer



def reproject(geom, from_proj=None, to_proj=None):
    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)


class Location:
    """A class to handel a location object. A given location (shapely polygon)
    can be compared against othe rlocations (polygons) to identify the spatial
    relationship between the two.
    
    parameters
    ----------
    shape : shapely.geometry.(Multi)Polygon | (Multi)Point | (Multi)linestring
        A shapely geometry object descirbing the spatial characteristics of a 
        locaiton.
    """
    def __init__(self, shape):        
        self.polygon = shape
        #print(self.polygon.geom_type)
        
    def is_target(self, test_location):
        """Tests to see if the test_location is the same location as the object,
        defined as intersection comprising >80% of both polygons.
        
        parameters
        ----------
        test_location : Location
        
        returns
        -------
        bool : True if the test location matches the target
        """

        test_area = test_location.polygon.area
        if test_location.polygon.area == 0:
            return False
        true_area = self.polygon.area
        intersection = test_location.polygon.intersection(self.polygon)
        
        test_coverage = intersection.area/test_area
        true_coverage = intersection.area/true_area
        return (test_coverage > 0.8) and (true_coverage > 0.8)
    
    def is_parent(self, test_location):
        """Tests whether test_location is a parent location to self.polygon
        (i.e. self.polygon is contained within test_location.polygon)
        
        parameters
        ----------
        test_location : Location
            The Location object to be tested against.
        returns
        -------
        Bool : True if test_location is parent of self.
        """
        test_poly = test_location.polygon
        
        if 'Polygon' not in test_poly.geom_type:
            return False
        if 'Polygon' not in self.polygon.geom_type:
            return False
        # test for full containment
        if self.contains_full_area(test_poly):
            return True
        # test for centroid containment (polygon contains centroid and >50% of area)
        if self.contains_centroid(test_poly)\
           and self.contains_half_area(test_poly):
            return True
        # otherwise not a child - return false
        return False
    
    def is_child(self, test_location):
        """Tests whether test_location is a child location of the object 
        location (i.e. self.polygon contains test_location.polygon)

        parameters
        ----------
        test_location : Location
            The location to be tested against.

        Raises
        ------
        AttributeError:'A Point object can not be a parent.
            If self.polygon is type Point

        Returns
        -------
        Bool : True if test_location is a child of self.polygon
        """
        location_copy = copy(self)
        
        return test_location.is_parent(location_copy)
    
    def contains_centroid(self, poly):
        """Tests if poly contains the location polygon centroid."""
        return poly.contains(self.polygon.centroid)
    
    def contains_full_area(self, poly):
        """Tests if the test polygon contians the full area of self.polygon."""
        # first need to fill any interior holes in the polygons
        if self.polygon.geom_type == 'MultiPolygon':
            mp1 = [Polygon(p) for p in list(self.polygon.geoms)]
            p1 = MultiPolygon([Polygon(p.exterior) for p in mp1])
        elif self.polygon.geom_type == 'Polygon':
            p1 = Polygon(self.polygon.exterior)
        else:
            p1 = self.polygon
        # and do the test poly
        if poly.geom_type == 'MultiPolygon':
            mp2= [Polygon(p) for p in list(poly.geoms)]
            p2 = MultiPolygon([Polygon(p.exterior) for p in mp2])
        elif poly.geom_type == 'Polygon':
            p2 = Polygon(poly.exterior)
        else:
            # a point or linestring can not be a parent
            return False
        # return true if p2 contains p1
        return p2.contains(p1)
    
    def contains_half_area(self, poly):
        """Tests if poly contains half the area of the location polygon."""
        intersection = poly.intersection(self.polygon)
        return intersection.area/self.polygon.area >= 0.5
        
    
    def is_adjacent(self, test_location):
        """Tests whether there the test locaiton is adjacent to the current 
        location; defined as either sharing an edge, or having some interesection
        with less than half"""
        # not adjacent if it is a parent or child location
        test_poly = test_location.polygon
        if test_poly.geom_type == 'Point':
            return False
        if self.is_child(test_location):
            return False
        if self.is_parent(test_location):
            return False
        # adjacent if there is a shared boundary
        if self.has_shared_boundary(test_poly):
            return True
        # also adjacent if there is an intersection without parent/child relation
        if test_poly.intersects(self.polygon):
            return True
        return False
    
    def min_distance(self, test_poly):
        polygon1_m = reproject(self.polygon, 'EPSG:4326', 'EPSG:26944')
        polygon2_m = reproject(test_poly, 'EPSG:4326', 'EPSG:26944')
        distance_in_meters = polygon1_m.distance(polygon2_m)
        return distance_in_meters
    
    def has_shared_boundary(self, poly):
        """tests if the test location has a shared boundary with the location"""
        return poly.intersection(self.polygon).geom_type == 'LineString'
    
    def is_crossing(self, test_linestring):
        return test_linestring.polygon.crosses(self.polygon)
    
    def get_relationship(self, test_location):
        if self.is_target(test_location):
            return 'TAR'
        if self.is_parent(test_location):
            return 'PAR' # note tagging is reversed - test_location is child
        if self.is_child(test_location):
            return 'CHI'
        if (self.is_adjacent(test_location)) and not (self.is_crossing(test_location)):
            return 'ADJ'
        if self.is_crossing(test_location):
            return 'CRO'
        # if no relationship return incidental
        return 'INC'

class Retagger:

    def __init__(self, tagged_df:pd.DataFrame, reference_df:pd.DataFrame):
        self.df = tagged_df
        self.reference_df = reference_df
        self.gazetteer = Gazetteer(polygon=True, addressdetails=False)
        try:
            self.search_history = self.load_search_history()
            print("Previous search history succesfully loaded")
        except:
            self.search_history = dict()

    def location_only_results(self, location_tags)->pd.DataFrame:
        """Returns only the results which are tagged as locations]

        parameters
        ----------
        location_tags : list(str)
            Tags used to identify locations in results dataframe.
        returns
        -------
        location_results : pd.DataFrame
            Results dataframe only including rows tagged with the location tags.
        """
        tags_regex = ''.join(f'{x}|' for x in location_tags)[:-1]
        location_results = self.df[self.df.Tag.str.contains(tags_regex, regex=True)]
        return location_results
    
    def get_true_location(self, sentence_number):
        """Uses the refence dataframe to get the name and coordinates of the 
        'true' location associated with the current sentence (i.e. the name
        of the wikipedia page the sentence was taken from).

        parameters
        ----------
        sentence_number : int
            The value in column 'Sentence #' in the results daatframe. This will
            refer to a specific row in the review_df dataframe.
        returns
        -------
        dict : {name:str, coords:(float,float))
            The name of the city and the (long, lat) coordinates.
        """
        reference = self.reference_df.loc[sentence_number]
        coords = reference.coordinates
        if isinstance(coords, str):
            coords = eval(coords)
        name = reference.city
        return {'name':name,'coords':coords}

    def get_true_polygon(self, true_name:str, true_coordinates:tuple)->list:   
        """Returns a polygon from Nominatim associated with the given name and
        containing the given coordinates. Note that if Nominatim only contains a 
        Point object for that location then the bounding box is returned.
        
        parameters
        ----------
        true_name : str
            The name of the location as given in the reference dataset.
        true_coordinates : tuple (float, float)
            The lat/lon coordinates of the location given in the refence data.
            
        returns
        -------
        polygon_matches : list [shpaely.geometry.Polygon/MultiPolygon]
            All polygons found matching the given name which contain the given 
            coordinate. 
        """
        query_data = self.gazetteer.query(true_name)
        polygon_matches = []
        for match in query_data.json():
            polygon = shape(match['geojson'])
            # if the geometry is point then use Bounding Box
            if polygon.geom_type=='Point':
                miny, maxy, minx, maxx = match['boundingbox']
                polygon = box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
                
            if polygon.contains(Point(true_coordinates[::-1])):
                polygon_matches.append(polygon)
                
        if len(polygon_matches)==0:
            return('Failed')
        elif len(polygon_matches)==1:
            return polygon_matches[0]
        else:
            return self.disambiguate_true_polygons(polygon_matches)

    
    def disambiguate_true_polygons(self, polygon_matches):
        """The basic way of disambiguating is to assume the more specific area
        is the true location. We'll do that by comparing the area covered by the 
        polygons"""
        areas = [p.area for p in polygon_matches]
        return polygon_matches[np.argmin(areas)]
        
            
    def retag(self, location_tags:list, offset:int, limit:int, save_period:int=2000):
        """This function builds phrases out of the tokens tagged with location
        tags in the results dataframe and uses nominatim to find the location
        reffered to by the word or phrase. It then builds a set of set of tags 
        depending on the spatial relationship between th etarget and the 
        location. 

        parameters
        ----------
        location_tags : list
            A list of the tags used in the results dataframe to idtentify a 
            location, or location-like, word. e.g. ['geo', 'gpe, 'org']. This
            does not need to include the 'B-' or 'I-' part of the tag. 
        offset : int
            If you need to start from part way through the dataset (for example 
            if a previous run failed due to connection etc) then this param will
            allow you to adjust the starting point for the retagging.
        limit : int
            Similar to offset, this will allow you to stop the process early if 
            required.
        save_period : int
            Allows you to set a period over which the search results will be
            saved. This is a large JSON file, so a short period will seriously
            slow down the process.

        updates
        -------
        self.results : pd.DataFrame
            Updates the 'Tag' column to provide relational tags. The old tags 
            (usually B-LOC etc) are now in the column 'old_tags'. Relational
            tags go in the 'Tag' column. 
        """
        # cut down to results tagged as locations as new Retagger object
        loc_only_results = Retagger(self.location_only_results(location_tags),
                                    self.reference_df)
        # add a column identifying sequential groups (phrases) in the dataframe
        loc_only_results.add_sequential_groups()
        loc_df = loc_only_results.df

        # loop over the phrases (sequential groups) in the location only data.
        new_tags = []
        tags_idx = []
        groups = loc_df.sequential_group.unique()
        j=0
        for group in tqdm(groups[offset:limit]):
            # get just the rows in that phrase/group
            phrase_df = loc_df[loc_df.sequential_group==group].copy()
            # get the review/sentence number that phrase belongs to
            sentence_num = phrase_df['Sentence #'].iloc[0]
            
            true_loc = self.get_true_location(sentence_num)
            # initialise new Phrase object and build phrase
            phrase = Phrase(token='', tag=None)
            for word, tag in phrase_df[['Word', 'Tag']].values:
                phrase.add_token(token=word, tag=tag)

            # first check for a target phrase (phrase == true_loc['name'])
            self_string = remove_parentheses(true_loc['name'].lower())
            test_string = remove_parentheses(phrase.text.lower())
            if self_string == test_string:
                new_tags += [prefix + 'TAR' for prefix in phrase.tags]
            # build the relational tags associated with the phrase
            else:
                new_tags += self.build_tags(phrase, true_loc)
                
            tags_idx += list(phrase_df.index.values)
            j += 1
            # save the search history every 100 iterations
            if save_period:
                if j%save_period == 0:
                    self.save_search_history()
            
        # update self.df with the new tags
        self.df['new_tag'] = 'O'
        self.df.loc[tags_idx, 'new_tag'] = new_tags
        if 'old_tag' in self.df.columns:
            self.df.drop('old_tag', axis=1, inplace=True)
        self.df.rename(mapper={'Tag':'old_tag', 'new_tag':'Tag'},
                       axis='columns', 
                       inplace=True)

    
    def add_sequential_groups(self):
        """Adds a new column to self.df which identifies seperate phrases in the
        text. It assumes that a single phrase will be a collection of tokens
        with sequential indexes. As such, data with indices [0,1,2,5,6,8,10,11]
        will be given sequential_group values [0,0,0,1,1,2,3,3]. This is desinged
        to work with data that has already been cut down to location only 
        tokens using self.location_only_results.
        
        updates
        -------
        self.df : pd.DataFrame
            Adds new column 'sequential_group' indicating the phrase each token
            belongs to.
        """
        groups = []
        current_group = 0
        for index in self.df.index:
            if len(groups) == 0:
                groups.append(current_group)
            elif index == self.df.index[self.df.index.get_loc(index) - 1] + 1:
                groups.append(current_group)
            else:
                current_group += 1
                groups.append(current_group)
        
        self.df['sequential_group'] = groups


    def build_tags(self, phrase, true_loc):
        """Builds the ['B-tar', 'B-inc', 'I-tar', 'I-inc'] tags for the data, based
        on proximity to the true location provided. This uses Nominatim to get all 
        the matches for a phrase, if any pass the proximity check then a 'tar' tag
        is given, otherwise a 'inc' tag is given.
        
        parameters
        ----------
        phrase : Phrase
            A Phrase object to build tags for.
        true_loc : dict {'name':str, 'coords':tuple}
            Name and location of place referenced by the text. 
              
        returns
        -------
        list : a list of tags for the provided phrase.
        """
        # get the true polygon associated the the full sentence
        
        true_poly = self.get_true_polygon(true_loc['name'], true_loc['coords'])
        # assume INC if location not found
        if true_poly == 'Failed':
            return [prefix + 'INC' for prefix in phrase.tags]
        true_location = Location(true_poly)
        # first check if phrase.text is the same string as the target:
        if phrase.text.lower() == true_loc['name'].lower():
            return [prefix + 'TAR' for prefix in phrase.tags]
        # check for city/county child relationships
        elif true_loc['name'].lower() == str(phrase.text + ' county').lower():
            return [prefix + 'CHI' for prefix in phrase.tags]
        # find all locations matching the phrase
        if phrase.text not in self.search_history.keys():
            try:
                query_data = self.search_gazetteer(phrase)
            except:
                print(f'Unable to search query "{phrase.text}"')
                return ['MISSING' for tag in phrase.tags]
        else:
            # otherwise use the stored data
            query_data = self.search_history[phrase.text]
        relations = []
        
        # get the relationship between each match and the true location
        for match in query_data:
            if isinstance(match, str):
                print(f'Unable to search query "{phrase.text}"')
                return ['MISSING' for tag in phrase.tags]
            try:
                test_poly = shape(match['geojson'])
            except KeyError:
                relations.append('NON')
                continue
            test_location = Location(test_poly)
            relations.append(true_location.get_relationship(test_location))
        non_inc_relations = set([x for x in relations if x != 'INC'])
        if set(relations) == {'INC'}:
            tag = 'INC'
        # if there is only one none INC relation, set as that.
        elif len(non_inc_relations)==1:
            tag = list(non_inc_relations)[0]

        # for conflicting matches, choose TAR then PAR then ADJ then CHI
        elif 'TAR' in relations:
            tag = 'TAR'
        elif 'PAR' in relations:
            tag = 'PAR'
        elif 'CRO' in relations:
            tag = 'CRO'
        elif 'ADJ' in relations:
            tag = 'ADJ'
        elif 'CHI' in relations:
            tag = 'CHI'
        else:
            tag = 'NON'
        return [prefix + tag for prefix in phrase.tags]
    
    def search_gazetteer(self, phrase):
        """Searches the OSM Nominatim API for a phrase, and adjusts the phrase
        if locations can not be found.
        
        parameters
        ----------
        phrase : Phrase object
            A data.Phrase object containg the search term.
            
        returns
        -------
        query_data : [dict]
            The JSON outputted by Nominatim given the search term. If no the 
            phrase is not matched to a location then an empty list is returned.
        """
        query_data = self.gazetteer.query(phrase.text).json()
        if not isinstance(query_data, list):
            query_data = [query_data]
        # also search for locations without common suffixes and prefixes
        # if no places are found, try removing unnecesary words
        if phrase.text != phrase.clean_text():
            query_data.extend(self.gazetteer.query(phrase.clean_text()).json())
        if phrase.text != phrase.remove_directions():
            query_data.extend(self.gazetteer.query(phrase.remove_directions()).json())
        if '-' in phrase.text:
            split_phrase = phrase.clean_text().split('-')
            for p in split_phrase:
                query_data.extend(self.gazetteer.query(p).json())
        # add the search results to the search history
        self.search_history[phrase.text] = query_data
        return query_data
    
    def save_search_history(self):
        with open('search_history.json', 'w') as j:
            json.dump(self.search_history, j)
            
    def load_search_history(self):
        with open('search_history.json', 'r') as j:
            search_history = json.load(j)       
        return search_history
            

    
    
def remove_parentheses(input_string):
    """Removes everything inside parenthesese from a string"""
    result = ""
    stack = []
    for char in input_string:
        if char == '(':
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()
            else:
                result += char
        elif not stack:
            result += char
    
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
    return result.replace('  ', ' ')