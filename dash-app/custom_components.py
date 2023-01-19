
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
from datetime import datetime
import pytz


# ========================
#         METHODS
# ========================
# The code below will help to specify different methods

# This method will create a "Result" Div


def generate_result_div(result_row):

    return html.Div(
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        width=2,
                        children=[
                            html.Div(
                                children=[
                                    html.Img(
                                        src=result_row.thumbnail_url,
                                        alt=f"Thumbnail for TheNeedleDrop video with the title '{result_row.title}'",
                                        width="naturalWidth",
                                        height="naturalHeight",
                                        style={"maxHeight": "200px"}
                                    )
                                ],
                                style={"textAlign": "center"}
                            )
                        ],
                        style={"paddingRight": "10px"}
                    ),

                    dbc.Col(
                        width=10,
                        children=[
                            html.Div(
                                children=[
                                    html.H3(
                                        html.A(
                                            href=result_row.url,
                                            target="_blank",
                                            children=[result_row.title]
                                        )
                                    ),
                                ],
                                style={"marginBottom": "16px"}
                            ),
                            dcc.Markdown(
                                children=[
                                    # {datetime.fromtimestamp(int(result_row.publish_date)/1000, tz=pytz.timezone("US/Eastern")).strftime("%x")
                                    f"""
                                    **Published on:** {result_row.publish_date}
                                    """
                                ], style={"marginBottom": "8px"}),
                            dcc.Markdown(
                                children=[
                                    f"""
                                    **Relevant Transcript Segment:** {result_row.segment_transcription}
                                    """
                                ],
                                style={"marginBottom": "8px"}
                            )
                        ]
                    )
                ]
            )
        ],
        style={"paddingTop": "20px", "paddingBottom": "20px"}
    )
