# This file contains various utilities that I'll use throughout the Dash app

# Import statements
import requests
import os
from requests.structures import CaseInsensitiveDict
import numpy as np
import json
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import math
import traceback
from time import time
from time import sleep
from IPython.display import display, Markdown


# This method will return a list of ndarrays, each representing text embeddings of
# the text in each index of the input_text_list list
def generate_embeddings(input_text_list, print_exceptions=False):
    # Get the OpenAI API key from the environment variables
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Build the API request
    url = "https://api.openai.com/v1/embeddings"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = "Bearer " + api_key
    data = """{"input": """ + json.dumps(input_text_list) + ""","model":"text-embedding-ada-002"}"""

    # Send the API request
    resp = requests.post(url, headers=headers, data=data)

    # If the request was successful, return ndarrays of the embeddings. Otherwise, return None objects
    if resp.status_code == 200:
        return [np.asarray(data_object['embedding']) for data_object in resp.json()['data']]
    else:
        if (print_exceptions):
            print(resp.json())
        return [None for txt in input_text_list]


# This method will generate the embedding for a single string
def generate_embedding(txt_input):
    return (generate_embeddings([txt_input])[0])


# This method will return the cosine similarity of two ndarrays
def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# This method will load in the tnd_data_df
def load_tnd_data_df():

    # Print some information
    print(f"\nLoading video data...\n")

    # Create a DataFrame containing all of the data scraped for each of the videos
    tnd_data_df_records = []
    for child_dir in tqdm(list(Path("../data/theneedledrop_scraping/").iterdir())):

        # Extract the video ID from the
        cur_video_id = child_dir.name

        # Load in the details.json file
        try:
            with open(f"../data/theneedledrop_scraping/{cur_video_id}/details.json", "r") as json_file:
                cur_details_dict = json.load(json_file)
        except:
            cur_details_dict = {}

        # Load in the transcription.json file
        try:
            with open(f"../data/theneedledrop_scraping/{cur_video_id}/transcription.json", "r") as json_file:
                cur_transcription_dict = json.load(json_file)
        except:
            cur_transcription_dict = {}

        # Load in the embedding
        try:
            with open(f"../data/theneedledrop_scraping/{cur_video_id}/whole_video_embedding.json", "r") as json_file:
                whole_video_embedding = json.load(json_file)
        except:
            whole_video_embedding = None

        # Load in the enriched details dictionary
        try:
            with open(f"../data/theneedledrop_scraping/{cur_video_id}/enriched_details.json", "r") as json_file:
                cur_enriched_details_dict = json.load(json_file)
        except:
            cur_details_dict = {}

        # Create a "record" for this video
        tnd_data_df_records.append({
            "video_id": cur_video_id,
            "details_dict": cur_details_dict,
            "transcription_dict": cur_transcription_dict,
            "whole_video_embedding": whole_video_embedding,
            "enriched_details_dict": cur_enriched_details_dict
        })

    # Now, we want to create a DataFrame from the tnd_data_df_records
    tnd_data_df = pd.DataFrame.from_records(tnd_data_df_records)

    # Making the embeddings ndarrays instead of lists
    tnd_data_df["whole_video_embedding"] = tnd_data_df["whole_video_embedding"].apply(
        lambda x: np.asarray(x) if x is not None else None)

    # Add a "transcription string" column
    tnd_data_df["transcription_str"] = tnd_data_df["transcription_dict"].apply(
        lambda x: x['text'] if 'text' in x else None)

    # Add a couple of columns indicating how long each of the transcriptions are
    tnd_data_df["transcription_length"] = tnd_data_df["transcription_str"].apply(
        lambda x: len(x) if x is not None else None)
    tnd_data_df["transcription_approx_tokens"] = tnd_data_df["transcription_str"].apply(
        lambda x: int(math.ceil(len(x) / 3.5)) if x is not None else None)

    # Add a couple of columns grabbing the title and URL of the video
    tnd_data_df["video_title"] = tnd_data_df["details_dict"].apply(lambda x: x['title'])
    tnd_data_df["video_url"] = tnd_data_df["video_id"].apply(lambda x: f"https://www.youtube.com/watch?v={x}")

    # Return the tnd_data_df
    return tnd_data_df


# This method will load in the segment_emb_df
def load_segment_emb_df():

    # Print some information
    print(f"\nLoading segment embeddings...\n")

    # Load in all of the JSON files containing the video segment embeddings
    video_segment_emb_dict = {}
    tnd_scraping_folder_path = Path("../data/theneedledrop_scraping/")
    for child_folder in tqdm(list(tnd_scraping_folder_path.iterdir())):
        if child_folder.is_dir():
            cur_video_id = child_folder.stem
            video_segment_emb_path = Path(
                f"../data/theneedledrop_scraping/{cur_video_id}/video_segment_embeddings.json")
            if (video_segment_emb_path.exists()):
                with open(video_segment_emb_path, "r") as json_file:
                    video_segment_emb_dict[cur_video_id] = json.load(json_file)

    # Loading all of the embedding dictionaries into a single DataFrame
    segment_emb_df_list = []
    for cur_video_id, segment_dict_list in video_segment_emb_dict.items():
        segmend_df = pd.DataFrame(segment_dict_list)
        segmend_df["video_id"] = cur_video_id
        segment_emb_df_list.append(segmend_df)
    segment_emb_df = pd.concat(segment_emb_df_list)

    # Remove all of the segments without embeddings
    segment_emb_df = segment_emb_df[segment_emb_df["embedding"].notna()].copy()

    # Return the DataFrame
    return segment_emb_df


# This method will perform "segment search"
def segment_search(search_txt,
                   tnd_data_df=None,
                   segment_emb_df=None):

    # Now, we want to create a DataFrame from the tnd_data_df_records
    if (tnd_data_df is None):
        tnd_data_df = load_tnd_data_df()

    if (segment_emb_df is None):
        segment_emb_df = load_segment_emb_df()

    # Print some information
    print(f"\nSearching the OpenAI API for '{search_txt}'\n")

    # Indicate the search string, and then generate an embedding based off of these
    search_txt_emb = generate_embedding(search_txt)

    # Print some information
    start_time = time()
    print(f"\nCalculating the similarity between the embedding for '{search_txt}' and each video's segment...\n")

    # Calculate the similarity between the segment embeddings and the search embedding
    segment_emb_sim_df = segment_emb_df.copy()
    segment_emb_sim_df["cosine_sim_to_search"] = segment_emb_sim_df["embedding"].apply(
        lambda x: cosine_sim(search_txt_emb, x))

    # Sort this DataFrame by the similarity to the search embedding
    sorted_segment_emb_sim_df = segment_emb_sim_df.sort_values(
        "cosine_sim_to_search", ascending=False).copy()

    grouped_sorted_segment_df = sorted_segment_emb_sim_df.groupby("video_id")
    top_segments_per_video = grouped_sorted_segment_df.apply(
        lambda x: x.sort_values("cosine_sim_to_search", ascending=False).head(10)).reset_index(
        drop=True).copy()
    top_videos_by_top_segments = top_segments_per_video.groupby("video_id")[
        "cosine_sim_to_search"].mean().reset_index().sort_values(
        "cosine_sim_to_search", ascending=False).merge(tnd_data_df, on="video_id").copy()

    # Print some information about how long the similarity calculation took
    end_time = time()-start_time
    print(f"The similarity calculation took {end_time:.2f} seconds.")

    return segment_emb_sim_df, top_videos_by_top_segments
