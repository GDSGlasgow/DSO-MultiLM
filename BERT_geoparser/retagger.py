<<<<<<< HEAD
# third party imports
import pandas as pd
from geopy.geocoders import Nominatim
from geopy import distance
from shapely.geometry import box, Point
from tqdm import tqdm
# local imports
from BERT_geoparser.data import Phrase


class Retagger:

    def __init__(self, results_df:pd.DataFrame):
        self.df = results_df

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
    
    def get_true_location(self, review_number, review_df):
        """Retrieves the 'true' location reffered to by the review. This will
        only work on the 'yelp review' data. Other geocoded data may require a 
        function.

        parameters
        ----------
        review_number : int
            The value in column 'Sentence #' in the results daatframe. This will
            refer to a specific row in the review_df dataframe.
        review_df : pd.DataFrame
            A pandas dataframe of yelp reviews with business locations 
            identified. 
        returns
        -------
        coords : (float, float)
            The (long, lat) coordinates referred to by the given review.
        """
        review = review_df.loc[review_number]
        coords = review.coordinates
        if isinstance(coords, str):
            coords = eval(coords)
        return coords

    
    def retag(self, location_tags, threshold, review_df):
        """This function builds phrases out of the tokens tagged with location
        tags in the results dataframe and uses nominatim to find the location
        reffered to by the word or phrase. It then builds a set of 'target' or
        'incidental' tags, depending on if the true location reffered to by the 
        review falls within the bound sof the identified location. 

        parameters
        ----------
        location_tags : list
            A list of the tags used in the results dataframe to idtentify a 
            location, or location-like, word. e.g. ['geo', 'gpe, 'org']. This
            does not need to include the 'B-' or 'I-' part of the tag. 
        threshold : float | str
            Either float or str('bbox'). If float valued then a location is 
            given the target ('tar') tag if it is less than the threshold 
            distance (in kilometers) from the true location. If 'bbox' is used 
            (recomended) then a token is given the 'tar' tag if the true location
            liew within the bounding box of any found locations. 
        review_df : pd.DataFrame
            This code only works if this is the yelp review dataset. The data 
            must have been linked to businesses to that there is a 'coordinates'
            column pointing to the true location of the business. See notebook
            NB2 for more details.

        updates
        -------
        self.results : pd.DataFrame
            Updates the 'Tag' column to vaue sof ['O', 'B-tar', 'B-inc', 
            'I-tar', 'I-inc'] indicating incidental and target locations. The 
            old 'Tag' column is preserved and renamed 'old_tag'. 
        """
        # check threshold type
        if threshold != 'bbox':
            raise('str valued threshold must be threshold="bbox".')
        
        # initialize new Nominatim object
        nom = Nominatim(user_agent='joseph.shingleton@glasgow.ac.uk')

        # cut down to results tagged as locations as new Retagger object
        loc_only_results = Retagger(self.location_only_results(location_tags))

        # add a column identifying sequential groups (phrases) in the dataframe
        loc_only_results.add_sequential_groups()
        loc_df = loc_only_results.df

        # loop over the phrases (sequential groups) in the location only data.
        new_tags = []
        tags_idx = []
        groups = loc_df.sequential_group.unique()
        for group in tqdm(groups):
            # get just the rows in that phrase/group
            phrase_df = loc_df[loc_df.sequential_group==group].copy()
            # get the review/sentence number that phrase belongs to
            sentence_num = phrase_df['Sentence #'].iloc[0]
            true_loc = self.get_true_location(sentence_num, review_df)
            # initialise new Phrase object and build phrase
            phrase = Phrase(token='', tag=None)
            for word, tag in phrase_df[['Word', 'Tag']].values:
                phrase.add_token(token=word, tag=tag)
            # build the 'tar' and 'inc' tags associated with the phrase
            new_tags += build_tags(phrase.text, phrase.tags, 
                                   threshold, true_loc, nom)
            tags_idx += list(phrase_df.index.values)
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
        
        self.df.loc[:, 'sequential_group'] = groups


def build_tags(phrase, phrase_tags, threshold, true_loc, nom):
    """Builds the ['B-tar', 'B-inc', 'I-tar', 'I-inc'] tags for the data, based
    on proximity to the true location provided. This uses Nominatim to get all 
    the matches for a phrase, if any pass the proximity check then a 'tar' tag
    is given, otherwise a 'inc' tag is given.
    
    parameters
    ----------
    phrase : str
        A string representation of a location within the text, e.g. 'new york'.
    phrase_tags : list
        A list of the current tags assigned to the phrase, e.g. 'B-geo',
        'I-org', 'B-gpe' etc. 
    threshold : float|str
        If a float is passed then the proximity check will return 'tar' if the 
        location is within the threshold distance (in KM) from true_loc. If
        'bbox' is passed (recomended), then the proximity check will return
        'tar' if true_loc is within the bounding box of the location returned
        by Nominatim. 
    true_loc : tuple(float, float)
        The true (long, lat) coordinates refferenced in the text. 
    nom : Nominatim
        A Nominatim object used to parse location phrases.
    
    returns
    -------
    list : a list of tags for the provided phrase.
    """
    if phrase.upper() == 'LA': # there is a weird interaction with 'LA' 
        phrase = 'Los Angeles'
    # ask Nominatim to parse the phrase
    try:
        matches = nom.geocode(query=phrase, exactly_one=False)

    # if this fails then assume its an incidental location
    except:
        matches = None
    if not matches:
        tag = 'inc'

    # otherwise run the proximity vcheck on all matched locations
    else:
        tag = proximity_check(matches, true_loc, threshold)
    # return the new tags
    return [prefix + tag for prefix in phrase_tags]
    
def proximity_check(matches, true_loc, threshold):
    """Checks the proximity of a set of geopy location objects with a true 
    location and a provided threshold.
    
    parameters
    ----------
    matches : list([geopy.Location])
        A list of geopy.location objects.
    true_location : tuple(float, float)
        A (long, lat) coordinate for the true location to be tested against
    threshold : float|str
        If a float is passed then the proximity check will return 'tar' if the 
        location is within the threshold distance (in KM) from true_loc. If
        'bbox' is passed (recomended), then the proximity check will return
        'tar' if true_loc is within the bounding box of the location returned
        by Nominatim. 

    returns
    -------
    str : 'tar'
        If the proximity test passes.
    str : 'inc'
        If the proximity test fails.
    """
    if threshold != 'bbox':
        match_locs = [(m.latitude, m.longitude) for m in matches]
        match_dist = [distance.distance(true_loc, ml).km for ml in match_locs]
        target = any([d < threshold for d in match_dist])
    else:
        shapely_bbx = [bbox_to_shapely(m.raw['boundingbox']) for m in matches]
        match_bbx = [box(*b) for b in shapely_bbx]
        target = any([m.contains(Point(true_loc[::-1])) for m in match_bbx])
        
    if target:
        return 'tar'
    else:
        return 'inc'


def bbox_to_shapely(nom_bbx):
    """Nominatim bbox is [miny, maxy, minx, maxx], shapely wants 
    [minx, miny, maxx, maxy]...
    """
    out = [float(c) for c in [nom_bbx[2], nom_bbx[0], nom_bbx[3], nom_bbx[1]]]
=======
# third party imports
import pandas as pd
from geopy.geocoders import Nominatim
from geopy import distance
from shapely.geometry import box, Point
from tqdm import tqdm
# local imports
from BERT_geoparser.data import Phrase


class Retagger:

    def __init__(self, results_df:pd.DataFrame):
        self.df = results_df

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
    
    def get_true_location(self, review_number, review_df):
        """Retrieves the 'true' location reffered to by the review. This will
        only work on the 'yelp review' data. Other geocoded data may require a 
        function.

        parameters
        ----------
        review_number : int
            The value in column 'Sentence #' in the results daatframe. This will
            refer to a specific row in the review_df dataframe.
        review_df : pd.DataFrame
            A pandas dataframe of yelp reviews with business locations 
            identified. 
        returns
        -------
        coords : (float, float)
            The (long, lat) coordinates referred to by the given review.
        """
        review = review_df.loc[review_number]
        coords = review.coordinates
        if isinstance(coords, str):
            coords = eval(coords)
        return coords

    
    def retag(self, location_tags, threshold, review_df):
        """This function builds phrases out of the tokens tagged with location
        tags in the results dataframe and uses nominatim to find the location
        reffered to by the word or phrase. It then builds a set of 'target' or
        'incidental' tags, depending on if the true location reffered to by the 
        review falls within the bound sof the identified location. 

        parameters
        ----------
        location_tags : list
            A list of the tags used in the results dataframe to idtentify a 
            location, or location-like, word. e.g. ['geo', 'gpe, 'org']. This
            does not need to include the 'B-' or 'I-' part of the tag. 
        threshold : float | str
            Either float or str('bbox'). If float valued then a location is 
            given the target ('tar') tag if it is less than the threshold 
            distance (in kilometers) from the true location. If 'bbox' is used 
            (recomended) then a token is given the 'tar' tag if the true location
            liew within the bounding box of any found locations. 
        review_df : pd.DataFrame
            This code only works if this is the yelp review dataset. The data 
            must have been linked to businesses to that there is a 'coordinates'
            column pointing to the true location of the business. See notebook
            NB2 for more details.

        updates
        -------
        self.results : pd.DataFrame
            Updates the 'Tag' column to vaue sof ['O', 'B-tar', 'B-inc', 
            'I-tar', 'I-inc'] indicating incidental and target locations. The 
            old 'Tag' column is preserved and renamed 'old_tag'. 
        """
        # check threshold type
        if threshold != 'bbox':
            raise('str valued threshold must be threshold="bbox".')
        
        # initialize new Nominatim object
        nom = Nominatim(user_agent='joseph.shingleton@glasgow.ac.uk')

        # cut down to results tagged as locations as new Retagger object
        loc_only_results = Retagger(self.location_only_results(location_tags))

        # add a column identifying sequential groups (phrases) in the dataframe
        loc_only_results.add_sequential_groups()
        loc_df = loc_only_results.df

        # loop over the phrases (sequential groups) in the location only data.
        new_tags = []
        tags_idx = []
        groups = loc_df.sequential_group.unique()
        for group in tqdm(groups):
            # get just the rows in that phrase/group
            phrase_df = loc_df[loc_df.sequential_group==group].copy()
            # get the review/sentence number that phrase belongs to
            sentence_num = phrase_df['Sentence #'].iloc[0]
            true_loc = self.get_true_location(sentence_num, review_df)
            # initialise new Phrase object and build phrase
            phrase = Phrase(token='', tag=None)
            for word, tag in phrase_df[['Word', 'Tag']].values:
                phrase.add_token(token=word, tag=tag)
            # build the 'tar' and 'inc' tags associated with the phrase
            new_tags += build_tags(phrase.text, phrase.tags, 
                                   threshold, true_loc, nom)
            tags_idx += list(phrase_df.index.values)
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
        
        self.df.loc[:, 'sequential_group'] = groups


def build_tags(phrase, phrase_tags, threshold, true_loc, nom):
    """Builds the ['B-tar', 'B-inc', 'I-tar', 'I-inc'] tags for the data, based
    on proximity to the true location provided. This uses Nominatim to get all 
    the matches for a phrase, if any pass the proximity check then a 'tar' tag
    is given, otherwise a 'inc' tag is given.
    
    parameters
    ----------
    phrase : str
        A string representation of a location within the text, e.g. 'new york'.
    phrase_tags : list
        A list of the current tags assigned to the phrase, e.g. 'B-geo',
        'I-org', 'B-gpe' etc. 
    threshold : float|str
        If a float is passed then the proximity check will return 'tar' if the 
        location is within the threshold distance (in KM) from true_loc. If
        'bbox' is passed (recomended), then the proximity check will return
        'tar' if true_loc is within the bounding box of the location returned
        by Nominatim. 
    true_loc : tuple(float, float)
        The true (long, lat) coordinates refferenced in the text. 
    nom : Nominatim
        A Nominatim object used to parse location phrases.
    
    returns
    -------
    list : a list of tags for the provided phrase.
    """
    if phrase.upper() == 'LA': # there is a weird interaction with 'LA' 
        phrase = 'Los Angeles'
    # ask Nominatim to parse the phrase
    try:
        matches = nom.geocode(query=phrase, exactly_one=False)

    # if this fails then assume its an incidental location
    except:
        matches = None
    if not matches:
        tag = 'inc'

    # otherwise run the proximity vcheck on all matched locations
    else:
        tag = proximity_check(matches, true_loc, threshold)
    # return the new tags
    return [prefix + tag for prefix in phrase_tags]
    
def proximity_check(matches, true_loc, threshold):
    """Checks the proximity of a set of geopy location objects with a true 
    location and a provided threshold.
    
    parameters
    ----------
    matches : list([geopy.Location])
        A list of geopy.location objects.
    true_location : tuple(float, float)
        A (long, lat) coordinate for the true location to be tested against
    threshold : float|str
        If a float is passed then the proximity check will return 'tar' if the 
        location is within the threshold distance (in KM) from true_loc. If
        'bbox' is passed (recomended), then the proximity check will return
        'tar' if true_loc is within the bounding box of the location returned
        by Nominatim. 

    returns
    -------
    str : 'tar'
        If the proximity test passes.
    str : 'inc'
        If the proximity test fails.
    """
    if threshold != 'bbox':
        match_locs = [(m.latitude, m.longitude) for m in matches]
        match_dist = [distance.distance(true_loc, ml).km for ml in match_locs]
        target = any([d < threshold for d in match_dist])
    else:
        shapely_bbx = [bbox_to_shapely(m.raw['boundingbox']) for m in matches]
        match_bbx = [box(*b) for b in shapely_bbx]
        target = any([m.contains(Point(true_loc[::-1])) for m in match_bbx])
        
    if target:
        return 'tar'
    else:
        return 'inc'


def bbox_to_shapely(nom_bbx):
    """Nominatim bbox is [miny, maxy, minx, maxx], shapely wants 
    [minx, miny, maxx, maxy]...
    """
    out = [float(c) for c in [nom_bbx[2], nom_bbx[0], nom_bbx[3], nom_bbx[1]]]
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
    return out