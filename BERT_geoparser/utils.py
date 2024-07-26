<<<<<<< HEAD
import numpy as np

def flatten(lists):
    "Flattens a list of lists into a single list."
    return [x for sublist in lists for x in sublist]

def convert(int32:np.int32)->int:
    """Utility function to convert int32 objects into regular int.
    parameters
    ----------
    value : np.int32
        numpy.int32 object to be converted.
    return : int
    """
    try:
        return int(int32)
    except:
        raise TypeError(f"Input of type {type(int32)} not able to be converted")
=======
import numpy as np

def flatten(lists):
    "Flattens a list of lists into a single list."
    return [x for sublist in lists for x in sublist]

def convert(int32:np.int32)->int:
    """Utility function to convert int32 objects into regular int.
    parameters
    ----------
    value : np.int32
        numpy.int32 object to be converted.
    return : int
    """
    try:
        return int(int32)
    except:
        raise TypeError(f"Input of type {type(int32)} not able to be converted")
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
