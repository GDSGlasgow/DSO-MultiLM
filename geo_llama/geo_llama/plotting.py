# third party imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go

"""A few functions to produce the output maps"""

def plot_map(json_locations, translate_cache):
    df = pd.DataFrame(json_locations)
    mapbox = get_mapbox(df)
    name_list = [translate_cache[n] for n in df['name'].tolist()]

    fig = go.Figure(go.Scattermapbox(
                customdata=name_list,
                lat=df['latitude'].tolist(),
                lon=df['longitude'].tolist(),
                mode='markers',
                marker=go.scattermapbox.Marker(size=15),
                hoverinfo="text",
                hovertemplate='<b>Name</b>: %{customdata}'
            ))
    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=mapbox)
    return fig    

def get_mapbox(df:pd.DataFrame)->dict:
    """Gets the parameters of a mapbox, given the geospatial data in the 
    dataframe.
    Args:
        df (pd.DataFrame): a dataframe with 'latitude' and 'longitude' cols.
    Returns:
        dict : the parameters of a mapbox which fully displays the data.
    """
    xmin, xmax, ymin, ymax = get_bounds(df)
    x_center = np.mean([xmin, xmax])
    y_center = np.mean([ymin, ymax])
    plot_center = go.layout.mapbox.Center(lat=y_center, lon=x_center)
    zoom = get_zoom(xmin, xmax, ymin, ymax)
    return dict(bearing=0, center=plot_center, pitch=0, zoom=zoom)

def get_bounds(df:pd.DataFrame)->tuple[float, float, float, float]:
    """retirves the spatial bounds of the data in the provided dataframe.
    Args:
        df (pd.DataFrame): a dataframe with 'latitude' and 'longitude' cols.
    Returns:
        tuple[float, float, float, float] : the spatial bounds as (xmin, xmax, 
        ymin, ymax).
    """
    ymin = np.amin([float(x) for x in df['latitude'].values])
    ymax = np.amax([float(x) for x in df['latitude'].values])
    xmin = np.amin([float(x) for x in df['longitude'].values])
    xmax = np.amax([float(x) for x in df['longitude'].values])
    return xmin, xmax, ymin, ymax

def get_zoom(xmin:float, xmax:float, ymin:float, ymax:float):
    """Produces an appropriate zoom level for agiven extent."""
    max_bound = max(abs(xmax-xmin), abs(ymax-ymin)) * 111
    zoom = 10 - np.log(max_bound)
    return zoom