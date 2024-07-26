<<<<<<< HEAD
# third-party imports
import pathlib
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


class Tokenizer:
    """An object to handle Bert tokenizer objects.

    Attributes
    ----------
    tokenizer_name : str
        the name of the tokenizer (e.g. 'bert-large-uncased')
    save_path : pathlib.Path
        the path to save the tokenizer data to

    methods
    -------
    save_tokenizer() -> None:
        Retrieves and saves the tokenizer given by tokenizer_name.

    load_tokenizer() -> BertWordPieceTokenizer:
        Loads the tokenizer saved in 'save_path'. If no tokenizer is found then 
        save_tokenizer() is called to retrieve and save a new tokenizer.

    set_tokenizer_name(size:str, cased:bool) -> str:
        sets tokenizer_name based on the values of 'size' and 'cased'.

    set_save_path(save_path:str)->pathlib.Path
        Sets the path to save the tokenizer data to. If no save path is 
        specified it will revert to tokenizer_name. The directory is also made 
        if it doesn't already exist.
    
    check_size(size:str) -> None:
        Raises an error if size is not 'base' or 'large'.    
    """

    def __init__(self, size:str='base', cased:bool=True, save_path:str=None):
        """
        Parameters
        ----------
        size : str
            The size of bert model used ('base' or 'large').
        cased : bool
            Set whether a case sensitive bert model is used (if yes cased=True).
        save_path : str
            Path to save tokenizer data. Uses tokenizer_name if not specified. 
            If the path does not exist then it will be made automatically. 
        """
        self.size = size
        self.cased = cased
        self.tokenizer_name = self.set_tokenizer_name(size, cased)
        self.save_path = self.set_save_path(save_path)
        

    def __repr__(self)->str:
        return self.tokenizer_name

    def __str__(self)->str:
        return self.tokenizer_name

    def save_tokenizer(self):
        """Retrieves and saves the tokenizer given by tokenizer_name.
        """
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.save_pretrained(self.save_path)

    def get_tokenizer(self)->BertWordPieceTokenizer:
        """
        Loads the tokenizer saved in 'save_path'. If no tokenizer is found then 
        save_tokenizer() is called to retrieve and save a new tokenizer.

        returns
        -------
        tokenizer : BertWordPieceTokenizer
            The tokenizer specified by self.tokenizer_name
        """
        vocab_file = pathlib.Path(self.save_path, 'vocab.txt')
        try:
            tokenizer = BertWordPieceTokenizer(str(vocab_file))
        except Exception:
            self.save_tokenizer()
            tokenizer = self.get_tokenizer()
        return tokenizer

    def set_tokenizer_name(self, size:str, cased:bool)->str:
        """sets tokenizer_name based on the values of 'size' and 'cased'.

        Parameters
        ----------
        size : str
            The size of the required bert tokenizer. Either 'base' or 'large'.
        cased : bool
            Sets whether a case sensitive bert tokenizer will be used.

        Returns
        -------
        tokenizer_name : str
            The name of the required bert tokenizer (e.g. 'bert-large-uncased').

        Raises
        ------
        Exception if size not in ['large', 'base']
        """
        self.check_size(size)
        # set cased/uncased:
        if cased:
            casing = 'cased'
        else:
            casing = 'uncased'
        # build name
        tokenizer_name = 'bert-' + size + '-' + casing
        return tokenizer_name

    def set_save_path(self, save_path:str=None)->pathlib.Path:
        """Sets the path to save the tokenizer data to. If no save
        path is specified it will revert to tokenizer_name. The directory
        is also made if it doesn't already exist. 
        parameters
        ----------
        save_path : str or None
            The name of the directory to save the tokenizer to.
        return save_path : pathlib.Path
            Path object pointing to tokenizer save directory. 
        """
        if not save_path:
            save_path = self.tokenizer_name.replace('-', '_') + '/'
        elif save_path[-1] != '/':
            save_path += '/'
        # make into pathlib.Path object and create directory
        save_path = pathlib.Path(save_path)
       # if not save_path.exists():
       #     save_path.mkdir(parents=True)
        return save_path

    def check_size(self, size):
        """Checks the value of size is either 'base' or 'large'.

        raises
        ------
        Exception : if size not in ['base', 'large']
        """
        if size not in ['base', 'large']:
            raise Exception("<size> should be 'base', or 'large'.")
=======
# third-party imports
import pathlib
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


class Tokenizer:
    """An object to handle Bert tokenizer objects.

    Attributes
    ----------
    tokenizer_name : str
        the name of the tokenizer (e.g. 'bert-large-uncased')
    save_path : pathlib.Path
        the path to save the tokenizer data to

    methods
    -------
    save_tokenizer() -> None:
        Retrieves and saves the tokenizer given by tokenizer_name.

    load_tokenizer() -> BertWordPieceTokenizer:
        Loads the tokenizer saved in 'save_path'. If no tokenizer is found then 
        save_tokenizer() is called to retrieve and save a new tokenizer.

    set_tokenizer_name(size:str, cased:bool) -> str:
        sets tokenizer_name based on the values of 'size' and 'cased'.

    set_save_path(save_path:str)->pathlib.Path
        Sets the path to save the tokenizer data to. If no save path is 
        specified it will revert to tokenizer_name. The directory is also made 
        if it doesn't already exist.
    
    check_size(size:str) -> None:
        Raises an error if size is not 'base' or 'large'.    
    """

    def __init__(self, size:str='base', cased:bool=True, save_path:str=None):
        """
        Parameters
        ----------
        size : str
            The size of bert model used ('base' or 'large').
        cased : bool
            Set whether a case sensitive bert model is used (if yes cased=True).
        save_path : str
            Path to save tokenizer data. Uses tokenizer_name if not specified. 
            If the path does not exist then it will be made automatically. 
        """
        self.size = size
        self.cased = cased
        self.tokenizer_name = self.set_tokenizer_name(size, cased)
        self.save_path = self.set_save_path(save_path)
        

    def __repr__(self)->str:
        return self.tokenizer_name

    def __str__(self)->str:
        return self.tokenizer_name

    def save_tokenizer(self):
        """Retrieves and saves the tokenizer given by tokenizer_name.
        """
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.save_pretrained(self.save_path)

    def get_tokenizer(self)->BertWordPieceTokenizer:
        """
        Loads the tokenizer saved in 'save_path'. If no tokenizer is found then 
        save_tokenizer() is called to retrieve and save a new tokenizer.

        returns
        -------
        tokenizer : BertWordPieceTokenizer
            The tokenizer specified by self.tokenizer_name
        """
        vocab_file = pathlib.Path(self.save_path, 'vocab.txt')
        try:
            tokenizer = BertWordPieceTokenizer(str(vocab_file))
        except Exception:
            self.save_tokenizer()
            tokenizer = self.get_tokenizer()
        return tokenizer

    def set_tokenizer_name(self, size:str, cased:bool)->str:
        """sets tokenizer_name based on the values of 'size' and 'cased'.

        Parameters
        ----------
        size : str
            The size of the required bert tokenizer. Either 'base' or 'large'.
        cased : bool
            Sets whether a case sensitive bert tokenizer will be used.

        Returns
        -------
        tokenizer_name : str
            The name of the required bert tokenizer (e.g. 'bert-large-uncased').

        Raises
        ------
        Exception if size not in ['large', 'base']
        """
        self.check_size(size)
        # set cased/uncased:
        if cased:
            casing = 'cased'
        else:
            casing = 'uncased'
        # build name
        tokenizer_name = 'bert-' + size + '-' + casing
        return tokenizer_name

    def set_save_path(self, save_path:str=None)->pathlib.Path:
        """Sets the path to save the tokenizer data to. If no save
        path is specified it will revert to tokenizer_name. The directory
        is also made if it doesn't already exist. 
        parameters
        ----------
        save_path : str or None
            The name of the directory to save the tokenizer to.
        return save_path : pathlib.Path
            Path object pointing to tokenizer save directory. 
        """
        if not save_path:
            save_path = self.tokenizer_name.replace('-', '_') + '/'
        elif save_path[-1] != '/':
            save_path += '/'
        # make into pathlib.Path object and create directory
        save_path = pathlib.Path(save_path)
       # if not save_path.exists():
       #     save_path.mkdir(parents=True)
        return save_path

    def check_size(self, size):
        """Checks the value of size is either 'base' or 'large'.

        raises
        ------
        Exception : if size not in ['base', 'large']
        """
        if size not in ['base', 'large']:
            raise Exception("<size> should be 'base', or 'large'.")
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
        