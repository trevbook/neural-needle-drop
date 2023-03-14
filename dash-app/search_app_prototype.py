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
import custom_components

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

# Run a SQL query to determine the minimum and maximum year for the year filter
min_max_year_df = custom_utils.query_to_df("""
SELECT
    MIN(YEAR(publish_date)) as min_year,
    MAX(YEAR(publish_date)) as max_year
FROM
    video_details 
""")
min_year_in_data = min_max_year_df.iloc[0].min_year
max_year_in_data = min_max_year_df.iloc[0].max_year

# Indicate the different video types that're in the data
video_types_to_labels = {
    "album_review": "Album Review",
    "ep_review": "EP Review",
    "mixtape_review": "Mixtape Review",
    "track_review": "Track Review",
    "weekly_track_roundup": "Weekly Track Roundup",
    "yunoreview": "Y U NO Review?",
    "vinyl_update": "Vinyl Update",
    "tnd_podcast": "TND Podcast",
    "misc": "Miscellaneous",
}


# ========================
#         LAYOUT
# ========================
# Below, we're going to set up the basic layout of the app

# Wrap the entire layout in a dbc.Container
app.layout = dbc.Container(
    children=[

        # Declare a couple of different Stores
        dcc.Store(id="query_emb", data=None),

        # Create a Modal
        dbc.Modal(
            id="transcription-modal",
            children=[
                dcc.Loading(
                    children=[
                        html.Div(
                            id="transcription-modal-content"
                        )
                    ]
                )
            ],
            is_open=False,
            size="xl"
        ),

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
                    style={"marginBottom": "10px"}
                ),

                # This Row will contain the filtering interface
                dbc.Row(
                    children=[

                        # This Column contains the year filter
                        dbc.Col(
                            width=4,
                            children=[
                                html.B("Year"),
                                html.Div(
                                    children=[
                                        dmc.RangeSlider(
                                            id="year-filter-range-slider",
                                            value=[min_year_in_data,
                                                   max_year_in_data],
                                            min=min_year_in_data,
                                            max=max_year_in_data,
                                            step=1,
                                            minRange=1
                                        )
                                    ],
                                    style={"position": "relative", "top": "25%"}
                                )
                            ]
                        ),

                        # This Column contains the video type filter
                        dbc.Col(
                            width=4,
                            children=[
                                html.B("Video Type"),
                                html.Div(
                                    children=[
                                        dmc.MultiSelect(
                                            placeholder="No filter set",
                                            id="video-type-filter-select",
                                            value=[],
                                            data=[
                                                {"value": key, "label": val} for key, val in video_types_to_labels.items()
                                            ],
                                        )
                                    ],
                                    style={"position": "relative", "top": "10%"}
                                )
                            ]
                        ),

                        # This Column contains the review score filter
                        dbc.Col(
                            width=4,
                            children=[
                                html.Div(
                                    id="review-score-filter-div",
                                    children=[
                                        html.B("Review Score"),
                                        html.Div(
                                            children=[
                                                dmc.RangeSlider(
                                                    id="review-score-filter-range-slider",
                                                    value=[0, 10],
                                                    min=0,
                                                    max=10,
                                                    step=1,
                                                    minRange=1
                                                )
                                            ],
                                            style={
                                                "position": "relative", "top": "25%"}
                                        )
                                    ],
                                    style={"visibility": "hidden"}
                                )
                            ]
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
                      Output("query_emb", "data")],
              inputs=[Input("trigger_search_button", "n_clicks")],
              state=[State("search_text_input", "value"),
                     State("year-filter-range-slider", "value"),
                     State("video-type-filter-select", "value"),
                     State("review-score-filter-range-slider", "value")])
def search_videos(trigger_search_button_n_clicks,
                  search_text_input,
                  year_filter_values, video_type_filter_values, review_score_filter_values):

    # If the user hasn't inputted a Search, we're not going to do anything
    if (search_text_input == "" or search_text_input is None or trigger_search_button_n_clicks == 0):
        raise PreventUpdate

    # Now that the data's been loaded, we're going to run the segment search
    top_scoring_video_details, query_emb = custom_utils.neural_tnd_video_search(
        search_text_input.strip(), year_filter_values,
        video_type_filter_values, review_score_filter_values,
        print_timing=True, return_emb=True)

    # Generate some Div to put into the Search results
    new_search_results_div = html.Div(
        children=[custom_components.generate_result_div(
            row) for row in top_scoring_video_details.itertuples()]
    )

    # Return the information about the data being loaded
    return new_search_results_div, query_emb.tolist()


# This callback will show the review-score-filter-div when "Album Review" is the only
# value selected in the video-type-filter-select
@app.callback(output=Output("review-score-filter-div", "style"),
              inputs=[Input("video-type-filter-select", "value")])
def hide_show_review_score_filter(video_type_filter_selection):
    if ("album_review" in video_type_filter_selection and len(video_type_filter_selection) == 1):
        return {"visibility": "visible", "height": "100%"}
    else:
        return {"visibility": "hidden"}


@app.callback(output=[Output("transcription-modal-content", "children"),
                      Output("transcription-modal", "is_open")],
              inputs=[
                  Input({"type": "video-link", "video_id": ALL}, "n_clicks")],
              state=[State("search_text_input", "value"),
                     State("query_emb", "data")], prevent_initial_call=True)
def update_and_open_modal(n_clicks,
                          user_query,
                          query_emb):

    
    if (False not in [x is None for x in n_clicks]):
        raise PreventUpdate

    try:
        # Determine which video was clicked from the callback context
        input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        video_id = json.loads(input_id)["video_id"]

        # Generate the content for the modal
        modal_content = custom_components.generate_transcription_display(
            video_id, user_query, query_emb)

        # Return the modal content (and open it)
        return modal_content, True

    except Exception as e:
        print(e)


# ========================
#           MAIN
# ========================
# The app itself is run below

if __name__ == "__main__":
    app.run_server(port=8000, debug=True)
