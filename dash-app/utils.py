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
import pandas as pd
import pinecone
import os
import mysql.connector
import traceback
import numpy as np
import math
from tqdm import tqdm
from time import time
import requests
from requests.structures import CaseInsensitiveDict
import json
from pathlib import Path
import traceback

# Set up the connection to the MySQL server
cnx = mysql.connector.connect(
    user='root', password=os.getenv("MYSQL_PASSWORD"), 
    host='localhost', database='neural-needle-drop')

# Create a cursor 
cursor = cnx.cursor()

# Initialize the Pinecone API connection
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Setting up the index 
pinecone_index = pinecone.Index("neural-needledrop-prototype")
pinecone_index.describe_index_stats()


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
def generate_embedding(txt_input, print_exceptions=False):
    return (generate_embeddings([txt_input], print_exceptions)[0])

def query_to_df(query, print_error=False):
    '''Query the active MySQL database and return results in a DataFrame'''

    # Try to return the results as a DataFrame
    try:
        # Execute the query
        cursor.execute(query)

        # Fetch the results 
        res = cursor.fetchall()

        # Return a DataFrame
        return pd.DataFrame(res, columns=[i[0] for i in cursor.description])

    # If we run into an Exception, return None
    except Exception as e:
        if (print_error):
            print(f"Ran into the following error:\n{e}\nStack trace:")
            print(traceback.format_exc())
        return None


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


def neural_tnd_video_search(search_str):

    """
    This method will run 'neural search' across all of TheNeedleDrop's videos, using 
    the methodology I'd established with Pinecone and MySQL. 

    It'll return two DataFrames: one containing the information about the top scoring 
    videos, and another containing the segments that scored the highest for this video.  
    """

    # Get the embedding for the search string
    search_str_emb = generate_embedding(search_str)

    # ================================================

    # Query the Pinecone index for the 5000 most similar 
    pinecone_results = pinecone_index.query(
        vector=search_str_emb.tolist(),
        filter={
            "embedding_type": "segment_chunk"
        },
        top_k=3000,
        include_metadata=True,
        namespace="video_embeddings"
    )

    # ================================================

    # Create a DataFrame from the Pinecone results 
    top_segment_matches_original_df = pd.DataFrame.from_records(
        [{"id": x.id, "score": x.score} | x.metadata
            for x in pinecone_results['matches']])

    grouped_sorted_segment_df = top_segment_matches_original_df.groupby("video_id")
    top_segment_matches_df = grouped_sorted_segment_df.apply(
        lambda x: x.sort_values("score", ascending=False).head(5)).reset_index(
        drop=True).copy()

    # Determine the average score across the different videos 
    avg_segment_sim_by_video_df = top_segment_matches_df.groupby("video_id")["score"].mean(numeric_only=True).reset_index().rename(
        columns={"score": "avg_segment_sim"}).sort_values("avg_segment_sim", ascending=False)

    median_segment_sim_by_video_df = top_segment_matches_df.groupby("video_id")["score"].median(numeric_only=True).reset_index().rename(
        columns={"score": "median_segment_sim"}).sort_values("median_segment_sim", ascending=False)

    segment_ct_by_video_df = top_segment_matches_df.groupby("video_id").count().reset_index().rename(
        columns={"id": "segment_ct"}).sort_values("segment_ct", ascending=False)[["video_id", "segment_ct"]]

    # Create the "scored_video_df", which tries to merge some degree of "relevance" and "frequency"
    scored_video_df = segment_ct_by_video_df.merge(avg_segment_sim_by_video_df, on="video_id")
    scored_video_df = scored_video_df.merge(median_segment_sim_by_video_df, on="video_id")
    scored_video_df["neural_search_score"] = scored_video_df["segment_ct"] * scored_video_df["avg_segment_sim"]
    scored_video_df = scored_video_df.sort_values("neural_search_score", ascending=False)

    # We'll also add in information about the most similar segment in each video 
    top_single_segments_per_video_df = top_segment_matches_df[top_segment_matches_df["video_id"].isin(
        list(scored_video_df.head(10)["video_id"]))]
    grouped_sorted_segment_df = top_single_segments_per_video_df.groupby("video_id")
    top_single_segments_per_video_df = grouped_sorted_segment_df.apply(
        lambda x: x.sort_values("score", ascending=False).head(1)).reset_index(
        drop=True).copy()

    # ================================================

    # This query will determine the information for the top videos
    top_scored_video_info_query_filter_str = " OR ".join([f'id="{row.video_id}"' for row in scored_video_df.head(10).itertuples()])
    top_scored_video_info_query = f"""
    SELECT
        *
    FROM
        video_details
    WHERE {top_scored_video_info_query_filter_str}"""

    # Execute the above query 
    top_scored_video_info_df = query_to_df(top_scored_video_info_query, print_error=True)

    # Merge in some of the scores 
    top_scored_video_info_df = top_scored_video_info_df.merge(scored_video_df, left_on="id", right_on="video_id").drop(
        columns=["video_id"]).sort_values("neural_search_score", ascending=False)

    # ================================================

    # Creating a "filter string" for the transcription query
    all_video_filter_str_list = []
    for row in top_single_segments_per_video_df.itertuples():
        segment_filter_str = " OR ".join([f"segment={num}" for num in list(range(int(row.start_segment), int(row.end_segment)+1))])
        all_video_filter_str_list.append(f"id='{row.video_id}' AND ({segment_filter_str})")
    transcription_filter_str = " OR ".join([f"({cur_vid_filter_str})" for cur_vid_filter_str in all_video_filter_str_list])

    # Crafting the transcription query 
    top_segment_transcriptions_query = f"""SELECT * FROM transcriptions WHERE {transcription_filter_str}"""

    # Executing the transcription query 
    top_segment_transcriptions_df = query_to_df(top_segment_transcriptions_query, print_error=True)

    # ================================================

    # Join together the individual segments to create segment chunks
    top_segment_chunk_per_video_df = top_segment_transcriptions_df.groupby("id")["text"].apply(list).reset_index()
    top_segment_chunk_per_video_df["text"] = top_segment_chunk_per_video_df["text"].apply(
        lambda seg_list: " ".join([seg.strip() for seg in seg_list]))
    top_segment_chunk_per_video_df = top_segment_chunk_per_video_df.rename(columns={"text": "segment_transcription"})
    top_segment_chunk_per_video_df = top_segment_chunk_per_video_df.merge(
        top_single_segments_per_video_df[["score", "video_id"]].rename(columns={"score": "top_segment_score"}), 
        left_on="id", right_on="video_id")

    # Merge these segment chunks back into the top_scored_video_info_df DataFrame
    top_scored_video_info_df = top_scored_video_info_df.merge(top_segment_chunk_per_video_df, on="id")

    # ================================================

    return top_scored_video_info_df