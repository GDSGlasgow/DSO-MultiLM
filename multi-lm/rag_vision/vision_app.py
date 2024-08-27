# standard library
from io import BytesIO
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party imports
import gradio as gr
import torch
import plotly.graph_objects as go
import numpy as np

from rag_vision.GPT4o_class import GPT4o

geo_locator = GPT4o(device="cuda" if torch.cuda.is_available() else "cpu")


# Function to handle the main processing logic
def process_image(uploaded_file, openai_api_key, num_nearest_neighbors, num_farthest_neighbors, context_text=None):
    if not openai_api_key:
        return "Please add your API key to continue.", None

    if uploaded_file is None:
        return "Please upload an image.", None

    # Use the set_image_app method to process the uploaded image
    geo_locator.set_image_app(
        file_uploader=uploaded_file,
        imformat='jpeg',
        use_database_search=True,  # Assuming you want to use the nearest/farthest neighbors
        num_neighbors=num_nearest_neighbors,
        num_farthest=num_farthest_neighbors,
        context_text=context_text
    )

    coordinates = geo_locator.get_location(
        OPENAI_API_KEY=openai_api_key,
        use_database_search=True  # Assuming you want to use the nearest/farthest neighbors
    )

    lat_str, lon_str = coordinates.split(',')
    lat_str = lat_str.strip("() ")
    lon_str = lon_str.strip("() ")
    latitude = float(lat_str)
    longitude = float(lon_str)

    # Generate the prediction map using Plotly
    prediction_map_fig = generate_plotly_map(
        latitudes=[latitude],
        longitudes=[longitude],
        titles=["Location Prediction"],
        colors=["red"],
        zoom_level=9
    )

    # Generate the nearest neighbor map using Plotly
    nearest_map_fig = None
    if geo_locator.neighbor_locations_array:
        nearest_latitudes, nearest_longitudes = zip(*geo_locator.neighbor_locations_array)
        nearest_map_fig = generate_plotly_map(
            latitudes=nearest_latitudes,
            longitudes=nearest_longitudes,
            titles=[f"({lat}, {lon})" for lat, lon in geo_locator.neighbor_locations_array],
            colors=["green"] * len(nearest_latitudes),
            zoom_level=6
        )

    # Generate the farthest neighbor map using Plotly
    farthest_map_fig = None
    if geo_locator.farthest_locations_array:
        farthest_latitudes, farthest_longitudes = zip(*geo_locator.farthest_locations_array)
        farthest_map_fig = generate_plotly_map(
            latitudes=farthest_latitudes,
            longitudes=farthest_longitudes,
            titles=[f"({lat}, {lon})" for lat, lon in geo_locator.farthest_locations_array],
            colors=["blue"] * len(farthest_latitudes),
            zoom_level=1
        )

    # Return the coordinates and Plotly figures directly
    return coordinates, prediction_map_fig, nearest_map_fig, farthest_map_fig


def generate_plotly_map(latitudes, longitudes, titles, colors, zoom_level):
    """
    Generate a Plotly map using Scattermapbox.

    Args:
        latitudes (list): List of latitudes for markers.
        longitudes (list): List of longitudes for markers.
        titles (list): List of titles (or hover text) for markers.
        colors (list): List of colors for markers.
        zoom_level (int): Zoom level for the map (higher value = closer zoom).

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object.
    """
    fig = go.Figure(go.Scattermapbox(
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(size=15, color=colors),
        text=titles,
        hoverinfo="text"
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=np.mean(latitudes), lon=np.mean(longitudes)),
            zoom=zoom_level
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    return fig


def main():
    # Gradio Interface
    with gr.Blocks() as vision_app:
        with gr.Row():
            with gr.Column():
                uploaded_file = gr.Image(label="Upload an image")
                openai_api_key = gr.Textbox(label="API Key", placeholder="xxxxxxxxx", type="password")

                with gr.Accordion("Advanced Options", open=False):
                    num_nearest_neighbors = gr.Number(label="Number of nearest neighbors", value=16)
                    num_farthest_neighbors = gr.Number(label="Number of farthest neighbors", value=16)

                submit = gr.Button("Submit")

            with gr.Column():
                status = gr.Textbox(label="Predicted Location")
                prediction_map_plot = gr.Plot(label="Predicted Location Point Map")
                with gr.Row():
                    nearest_map_plot = gr.Plot(label="Similar Location Points Map")
                    farthest_map_plot = gr.Plot(label="Dissimilar Location Points Map")

        submit.click(
            process_image,
            inputs=[
                uploaded_file,
                openai_api_key,
                num_nearest_neighbors,
                num_farthest_neighbors
            ],
            outputs=[status, prediction_map_plot, nearest_map_plot, farthest_map_plot]
        )

    vision_app.launch()


if __name__ == '__main__':
    # Initialize the GPT4v2Loc object
    geo_locator = GPT4o(device="cuda" if torch.cuda.is_available() else "cpu")
    main()
