
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
from dash import html, Input, Output, callback, State, MATCH, ALL, callback_context, dcc
import utils as custom_utils
from datetime import datetime
import pytz
import pinecone
from transcription_display import TranscriptionDisplay
import dash_player as dp

# Initialize the Pinecone API connection
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Setting up the index
pinecone_index = pinecone.Index("neural-needledrop-prototype")
pinecone_index.describe_index_stats()


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
                                        html.Button(
                                            className="link-button",
                                            id={"type": "video-link",
                                                "video_id": result_row.id},
                                            children=[result_row.title]
                                        ),
                                        # html.A(
                                        #     href=result_row.url,
                                        #     target="_blank",
                                        #     children=[result_row.title]
                                        # )
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


def generate_transcription_display(
    cur_video_id,
    user_query,
    user_search_query_emb
):

    def segment_text_link(segment_text, seek, similarity_to_query_str=0.5):

        return html.Button(
            className="astext",
            children=[segment_text],
            id={"type": "segment_text_button",
                "seek": seek}
        )

    # Query the Pinecone index for all of the 
    pinecone_results = pinecone_index.query(
        vector=user_search_query_emb,
        filter={
            "embedding_type": "segment_chunk",
            "video_id": cur_video_id
        },
        top_k=10000,
        include_metadata=True,
        namespace="video_embeddings"
    )

    # Create a DataFrame from the Pinecone results 
    top_segment_matches_original_df = pd.DataFrame.from_records(
        [{"id": x.id, "score": x.score} | x.metadata
            for x in pinecone_results['matches']])

    # Collect the relevant data from the transcriptions table
    cur_video_transcription_df = custom_utils.query_to_df(f"""SELECT * FROM transcriptions WHERE id="{cur_video_id}" AND segment != -1""")

    # Creating the segment_chunk_df
    segment_chunk_df = top_segment_matches_original_df[["start_segment", "end_segment", "score"]].copy()

    # Creating the segment_info_df
    segment_info_df = cur_video_transcription_df[["segment", "start_time", "text"]].rename(
        columns={"start_time": "seek"}).copy()

    return dbc.Container(
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
                        id={"type": "dash-player", "video_id": cur_video_id},
                        url=f"https://youtu.be/{cur_video_id}",
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
                    dcc.Markdown(f"""
                    ### **Query:** {user_query}
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
                    TranscriptionDisplay(segment_chunks=segment_chunk_df.to_dict(orient="records"), segment_info=segment_info_df.to_dict(orient="records"), id={"type": "transcription-display", "video_id": cur_video_id})
                ]
            )
        ],
        fluid=True
    )

# This callback will change the position of the video based on the seek value
@callback(Output({"type": "dash-player", "video_id": MATCH}, "seekTo"),
          Input({"type": "transcription-display", "video_id": MATCH}, "seek"))
def update_player_seek(seek):

    if (seek is not None):
        return seek
    else:
        return 0


