# This is the main entrypoint into the Viewing Prototype. This app was written by Trevor Hubbard

# ========================
#          SETUP
# ========================
# The code below helps to set up the rest of the app

# Some import statements
import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate
from dash import html, Input, Output, State, MATCH, ALL, callback_context, dcc
import dash_player as dp
from utils import query_to_df
import json

# Declaring the app and server objects that'll be used throughout the app
app = dash.Dash(__name__,
                external_stylesheets=["https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                                      "https://fonts.googleapis.com/css2?family=Rubik:wght@300;400&display=swap",
                                      dbc.themes.BOOTSTRAP])

# Adding a title to the Dash app
app.title = "Video Viewing Prototype"

# ========================
#       LOADING DATA
# ========================
# The cells below will load in necessary data for the Dash app

# Spoof the input, which is a video ID
input_video_id = "__PxaWntvhg"

# Load the transcription data for this input video
input_video_transcription_df = query_to_df(f"""
SELECT * FROM transcriptions
WHERE id='{input_video_id}' AND segment != -1
""")

# ========================
#    CUSTOM COMPONENTS
# ========================
# I'm going to declare a couple of custom components below; I'll eventually
# move these into a different file, but they're solid for here now.


def segment_text_link(segment_text, seek, similarity_to_query_str=0.5):

    return html.Button(
        className="astext",
        children=[segment_text],
        id={"type": "segment_text_button",
            "seek": seek}
    )


# ========================
#         LAYOUT
# ========================
# Below, we're going to set up the basic layout of the app
# Wrap the entire layout in a dbc.Container


text = """
This is a sentence. This is another sentence.
"""

sentences = text.split(".")

app.layout = dbc.Container(
    children=[

        dcc.Location(
            id="location"
        ),

        dbc.Row(
            children=[
                "Video Information Row"
            ]
        ),

        dbc.Row(
            children=[
                dp.DashPlayer(
                    id="dash-player",
                    url=f"https://youtu.be/{input_video_id}",
                    controls=True,
                    width="100%",
                    height="600px",
                    playing=True,
                    seekTo=0
                ),
            ]
        ),

        dbc.Row(
            children=[
                "Query Information Row"
            ]
        ),

        dbc.Row(
            children=[
                html.Div(
                    children=[
                        segment_text_link(row.text, row.start_time) for
                        row in input_video_transcription_df.itertuples()
                    ],
                    style={"display": "inline-block"}
                )
            ]
        )
    ],
    fluid=True
)


# ========================
#        CALLBACKS
# ========================
# All of the callbacks for the file will be below

# This callback will change the position of the video based on the seek value
@app.callback(Output("dash-player", "seekTo"),
              Input({"type": "segment_text_button", "seek": ALL}, "n_clicks"))
def update_player_seek(n_clicks):

    try:
        triggering_seek_info_dict_str = dash.callback_context.triggered[0]['prop_id'].split(".n_clicks")[
            0]
        if (triggering_seek_info_dict_str == "."):
            return 0
        else:
            triggering_seek_info_dict = json.loads(
                triggering_seek_info_dict_str)
            new_seek_val = triggering_seek_info_dict["seek"]
            return new_seek_val
    except Exception as e:
        print(e)
        raise PreventUpdate


# ========================
#           MAIN
# ========================
# The app itself is run below
if __name__ == "__main__":
    app.run_server(port=8005, debug=True)
