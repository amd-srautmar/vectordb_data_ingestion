from llmcore.cms import sharepointutils as sp
from llmcore.cms import cmfunctions
import llmcore.cms.configgen as gen
import sys
import pandas as pd

#Get Sharepoint & Filepath Config
my_config = gen.load_config("cfg.toml")
SHAREPOINT_SITE_URL = my_config.sharepoint_config.sharepoint_site_url
SHAREPOINT_FOLDER = my_config.sharepoint_config.sharepoint_folder
relative_url = SHAREPOINT_SITE_URL.replace("https://amdcloud.sharepoint.com","")+SHAREPOINT_FOLDER
USERNAME = my_config.sharepoint_config.username
PASSWORD = my_config.sharepoint_config.password

filepath = my_config.sharepoint_config.local_filepath
local_dir = sys.path[0]
path = local_dir+filepath

load_dt = "2023-01-01"

# column names for the vector dataframe
column_dict_prepare={
    "column1": "name",
    "column2": "url",
    "column3": "modified_dt",
    "column4": "index",
    "column5": "text",
    "column6": "vector",
    "column7": "n_tokens",
    "column8": "chunk_id",
    "column9": "load_dt",
    "column10": "source",
    "column11": "title"
  }

local_data_dir = sys.path[0]+"/data/"
EMBEDDING_ENDPOINT =  "embeddings"
EMBEDDING_DEPLOYMENT_ID = my_config.llm_service_config.embedding_engine
API_KEY = my_config.llm_service_config.api_proxy_key
API_PROXY_URL = my_config.llm_service_config.api_proxy_url
CHUNK_SIZE = 2000
CHUNK_OVERLAP_SIZE = 50

def perform_sharepoint_data_pull(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD, filepath, incremental=False, verbose=True):
    """Function to pull data files from sharepoint folder. Datafiles can be .pdf, .docx, .pptx, .txt, .mp4.

    Args:
        SHAREPOINT_SITE_URL (_type_): URL to the Sharepoint site
        relative_url (_type_): Relative URL to the folder containing the data files
        USERNAME (_type_): Sharepoint username
        PASSWORD (_type_): Sharepoint password
        filepath (_type_): Local filepath to store the data files
        incremental (bool, optional): Defaults to False.
        verbose (bool, optional): Defaults to True.
    """
    if incremental:
        print("This is in the inc=True section. The incremental value = " + str(incremental))
        df = pd.read_csv('files_df.csv', index_col=False)
        sp.scrape_sharepoint(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD, filepath, df, incremental, verbose=verbose)
        df = sp.get_file_df(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD)
        
    else:
        print("This is in the inc=False section. The incremental value = " + str(incremental))
        print("relative_url", relative_url)
        sp.scrape_sharepoint(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD, filepath, files_df=None, incremental=False, verbose=verbose)
        df = sp.get_file_df(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD)
        
        
    df.to_csv('files_df.csv')
    return df

# -------------------------------------------------------------------------
# Incremental Refresh
# -------------------------------------------------------------------------
#perform_sharepoint_data_pull(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD, filepath, incremental=True, verbose=True)

# -------------------------------------------------------------------------
# Full Refresh
# -------------------------------------------------------------------------
data_df = perform_sharepoint_data_pull(SHAREPOINT_SITE_URL, relative_url, USERNAME, PASSWORD, filepath, incremental=False, verbose=True)

# Prepare vector data to upload into Weaviate
cloud_df_with_vectors = sp.weaviate_vector_data_preparation(data_df, load_dt, column_dict_prepare, local_data_dir, 
                                                            EMBEDDING_DEPLOYMENT_ID,API_PROXY_URL, API_KEY,
                                                            CHUNK_SIZE, CHUNK_OVERLAP_SIZE)
cloud_df_with_vectors.to_csv('cloud_df_with_vectors.csv')

# Load vector data to Weaviate
WEAVIATE_CLASS = my_config.query_config.vector_classes.weaviate_class
WEAVIATE_CLIENT = my_config.llm_service_config.vector_db_url
column_dict_load_kb = {
"column1": "text",
"column2": "url",
"column3": "source",
"column4": "n_tokens"
}
BATCH_SIZE = 150
cmfunctions.data_load_weaviate(WEAVIATE_CLASS, cloud_df_with_vectors, WEAVIATE_CLIENT,
                   column_dict_load_kb, BATCH_SIZE)