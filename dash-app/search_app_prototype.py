# This is the main entrypoint into the Neural Needledrop App. This app was written by Trevor Hubbard

# ========================
#          SETUP
# ========================
# The code below helps to set up the rest of the app

# Some import statements
import dash
import json
import os
import pandas as pd
from time import sleep
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash import html, Input, Output, State, MATCH, ALL, callback_context, dcc
import utils as custom_utils

# Declaring the app and server objects that'll be used throughout the app
app = dash.Dash(__name__,
                external_stylesheets=["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                                      "https://fonts.googleapis.com/css2?family=Rubik:wght@300;400&display=swap",
                                      dbc.themes.BOOTSTRAP])

# Adding a title to the Dash app
app.title = "Neural Needledrop"

# ========================
#       LOADING DATA
# ========================
# The cells below will load in necessary data for the

# We'll start without the tnd_data / embeddings loaded
tnd_data_df = None
segment_emb_df = None

# ========================
#         LAYOUT
# ========================
# Below, we're going to set up the basic layout of the app

# Wrap the entire layout in a dbc.Container
app.layout = dbc.Container(
    children=[

        # Declare a couple of different Stores
        dcc.Store(id="data_loaded_store", data=False),

        # We're going to wrap everything AGAIN - this time in a Div
        html.Div(
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            width=10,
                            children=[
                                html.H1("Neural Needledrop")
                            ]
                        ),
                        dbc.Col(
                            width=2,
                            children=[
                                html.H4(
                                    id="data_loaded_status_div",
                                    children=[
                                        "Data Not Loaded."
                                    ],
                                    style={"textAlign": "right", "color": "red"}
                                )
                            ]
                        )
                    ],
                    style={"marginBottom": "20px"}
                ),

                # This Row will contain the Search interface
                dbc.Row(
                    children=[

                        # This Column will contain the TextInput that acts as the Search button
                        dbc.Col(
                            width=10,
                            children=[
                                dmc.TextInput(
                                    style={"width": "100%"},
                                    radius="md",
                                    id="search_text_input",
                                )
                            ],
                        ),

                        # This Column will contain the Button that triggers the Search
                        dbc.Col(
                            width=2,
                            children=[
                                dmc.Button(
                                    children=["Search"],
                                    radius="lg",
                                    fullWidth=True,
                                    color="Blue",
                                    id="trigger_search_button"
                                )
                            ],
                        )
                    ],
                    style={"marginBottom": "30px"}
                ),

                # This Row will contain the Search results
                dbc.Row(
                    children=[

                        # Wrap the Search Results Div in a dmc.Loader
                        dbc.Spinner(
                            children=[
                                html.Div(
                                    id="search_results_div",
                                    children=[
                                        dmc.Alert(
                                            children=[
                                                "Search for a video using whatever phrase you want, and then hit 'Search'."
                                            ],
                                            color="blue"
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    style={}
                )
            ],
            style={"margin": "10px"}
        )
    ],
    fluid=True,
    style={}
)


# ========================
#        CALLBACKS
# ========================
# All of the callbacks for the file will be below


# This callback will trigger the video search when the user clicks the "search" button
@app.callback(output=[Output("search_results_div", "children"),
                      Output("data_loaded_status_div", "children"),
                      Output("data_loaded_status_div", "style"),
                      Output("data_loaded_store", "data")],
              inputs=[Input("trigger_search_button", "n_clicks")],
              state=[State("search_text_input", "value"),
                     State("data_loaded_store", "data")])
def search_videos(trigger_search_button_n_clicks,
                  search_text_input,
                  data_is_loaded):
    # Indicate that we're using a global variable
    global tnd_data_df
    global segment_emb_df

    # If the user hasn't inputted a Search, we're not going to do anything
    if (search_text_input == "" or search_text_input is None or trigger_search_button_n_clicks == 0):
        raise PreventUpdate

    # Handle the search
    sleep(5)

    # If the data hasn't been loaded, we're going to load it
    if (not data_is_loaded):
        # Load the data
        tnd_data_df = custom_utils.load_tnd_data_df()
        segment_emb_df = custom_utils.load_segment_emb_df()

    # Now that the data's been loaded, we're going to run the segment search
    segment_emb_sim_df, top_videos_by_top_segments_df = custom_utils.segment_search(search_text_input.strip(),
                                                                                    tnd_data_df,
                                                                                    segment_emb_df)

    # Save both of these results; this'll help for testing purposes
    print("\nSaving the results for use as test data...\n")
    segment_emb_sim_df.head(10).to_json("./../test_segment_emb_sim_df.json", orient="records", indent=2)
    top_videos_by_top_segments_df.head(100).to_json("./../test_top_videos_by_top_segments_df.json", orient="records", indent=2)

    # Print the results to ensure that things are working correctly
    print(f"\n{top_videos_by_top_segments_df.head(3)}")
    print(f"\n{top_videos_by_top_segments_df.columns}")

    # Generate some Div to put into the Search results
    search_results_markdown_version = """"""
    for idx, vid_row in enumerate(list(top_videos_by_top_segments_df.head(5).itertuples())):
        cur_vid_markdown = f"""
        Result #{idx+1}: {vid_row.video_title}
        (Cosine Similarity: {vid_row.cosine_sim_to_search})
        
        """
        search_results_markdown_version += cur_vid_markdown
    new_search_results_div = html.Div(children=[search_results_markdown_version])

    # Generate some information about the data being loaded
    new_data_loaded_div_children = "Data Loaded"
    new_data_loaded_div_style = {"textAlign": "right", "color": "green"}

    # Return the information about the data being loaded
    return new_search_results_div, new_data_loaded_div_children, new_data_loaded_div_style, True


# ========================
#           MAIN
# ========================
# The app itself is run below

if __name__ == "__main__":
    app.run_server(port=8000, debug=True)
