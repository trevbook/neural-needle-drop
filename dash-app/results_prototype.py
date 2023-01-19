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
from custom_components import generate_result_div

# Declaring the app and server objects that'll be used throughout the app
app = dash.Dash(__name__,
                external_stylesheets=["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                                      "https://fonts.googleapis.com/css2?family=Rubik:wght@300;400&display=swap",
                                      dbc.themes.BOOTSTRAP])

# Adding a title to the Dash app
app.title = "Search Results Prototype"

# ========================
#       LOADING DATA
# ========================
# The cells below will load in necessary data for the Dash app

# Load in the test
test_data_df = pd.read_json("./../data/test_segment_emb_sim_df.json")


# ========================
#         LAYOUT
# ========================
# Below, we're going to set up the basic layout of the app
# Wrap the entire layout in a dbc.Container

app.layout = dbc.Container(
    children=[
        html.Div(
            children=[
                generate_result_div(row) for
                row in test_data_df.itertuples()
            ]
        )
    ],
    fluid=True
)


# ========================
#        CALLBACKS
# ========================
# All of the callbacks for the file will be below


# ========================
#           MAIN
# ========================
# The app itself is run below

if __name__ == "__main__":
    app.run_server(port=8005, debug=True)
