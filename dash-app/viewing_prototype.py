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
from dash import html, Input, Output, State, MATCH, ALL, callback_context, callback, dcc
import dash_player as dp
from utils import query_to_df
import json
from transcription_display import TranscriptionDisplay

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

# Indicating what the user's search query is 
user_search_query = "Man on the moon, spaceship going to the firey depths of Hell"

# Spoof the input, which is a video ID
input_video_id = "J_LvLhFq2IU"

# Load the transcription data for this input video
input_video_transcription_df = query_to_df(f"""
SELECT * FROM transcriptions
WHERE id='{input_video_id}' AND segment != -1
""")

# Open the files with the test information
with open("data/test_segment_chunk_df.json", "r") as json_file:
    segment_chunks = json.load(json_file)
with open("data/test_segment_info_df.json", "r") as json_file:
    segment_info = json.load(json_file)

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
                    playing=False,
                    seekTo=0
                ),
            ]
        ),

        dbc.Row(
            children=[
                dcc.Markdown(f"""
                ### **Query:** {user_search_query}
                """)
            ]
        ),

        dbc.Row(
            children=[
                # html.Div(
                #     children=[
                #         segment_text_link(row.text, row.start_time) for
                #         row in input_video_transcription_df.itertuples()
                #     ],
                #     style={"display": "inline-block"}
                # )
                TranscriptionDisplay(segment_chunks=segment_chunks, segment_info=segment_info, id="transcription-display")
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
@callback(Output("dash-player", "seekTo"),
              Input("transcription-display", "seek"))
def update_player_seek(seek):

    if (seek is not None):
        return seek
    else:
        return 0


# ========================
#           MAIN
# ========================
# The app itself is run below
if __name__ == "__main__":
    app.run_server(port=8005, debug=True)
