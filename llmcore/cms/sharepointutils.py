from office365.runtime.auth.user_credential import UserCredential  # NOT USED
from office365.runtime.http.request_options import RequestOptions  # NOT USED
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File  # NOT USED
from office365.sharepoint.folders.folder import Folder  # NOT USED
import sys
import os
import pandas as pd
import re

import requests
from requests_ntlm import HttpNtlmAuth
from llmcore.cms import parsers,cmfunctions
from datetime import datetime
import uuid
import tiktoken

def create_sharepoint_client_context(sharepoint_url:str, username:str, password: str):
    """ Use the Site Url"""
    try:
        ctx = ClientContext(sharepoint_url).with_user_credentials(username, password)  
        return ctx
    except Exception as e:
        print(str(e)) 
        
def get_subfolders(ctx, parent_folder: str):
    myfldr = ctx.web.get_folder_by_server_relative_url(parent_folder).get().execute_query()
    ctx.load(myfldr, ["Folders"]).execute_query()
    subfolders = myfldr.folders
    return subfolders
    
def get_files_from_folder(ctx, folder_url: str):
    """ use the relative url"""
    try:
        myfldr = ctx.web.get_folder_by_server_relative_url(folder_url).get().execute_query()
        ctx.load(myfldr, ["Files"]).execute_query()
        fileobjs = myfldr.files
        return fileobjs
    except Exception as e:
        print(str(e)) 

def get_urls_for_all_subfolders(ctx, to_map): 
    ''' to_map is the Sharepoint folder to recurse through'''
    folder_handler_list = []
    
    def enum_folder(parent_folder):
        folders = get_subfolders(ctx, parent_folder)
        if len(folders) == 0:
            pass
        else:    
            print("Subfolders exist. Recursing further.")
    
            for folder in folders:
                print("Processing - - - "+folder.name)
                url = folder.serverRelativeUrl
                folder_handler_list.append(url)
                enum_folder(url)
    
    root_folder = ctx.web.get_folder_by_server_relative_url(to_map).get().execute_query()
    enum_folder(root_folder.serverRelativeUrl)
    print(f"\nMapping complete. + {len(folder_handler_list)} folder/s found.")
    return folder_handler_list

def get_all_files_from_directory(ctx, parent_folder: str):
    all_subfolder_urls = get_urls_for_all_subfolders(ctx, parent_folder)
    all_subfolder_urls.append(parent_folder)
    all_files = [] #instantiate an empty list to extend with all the file objects
    for url in all_subfolder_urls:
        files = get_files_from_folder(ctx, url)
        all_files.extend(files)
    return all_files #returns a list of file objects

def download_files(fileobjs, ctx, filepath:str, relative_url:str, verbose=False):
    local_dir = sys.path[0]
    newpath = local_dir+filepath
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for file in fileobjs:
        filename = file.name
        download_path = local_dir+filepath+'/'+filename
        file_url = file.serverRelativeUrl
        print("file_url", file_url)
        with open(download_path, "wb") as local_file:
            file = ctx.web.get_file_by_server_relative_url(file_url).download(local_file).execute_query()
        if verbose:
            print("[Ok] file has been downloaded into: {0}".format(download_path))

def download_files_cloud(ctx, file_df, folder_path):
    """Function to download files from sharepoint folder.

    Args:
        ctx (object): Sharepoint client context
        file_df (DataFrame): Input dataframe
        folder_path (str): Folder path to store downloaded files
        sharepoint_domain (str): Sharepoint domain url
    """
    for file_name, file_url in zip(file_df["name"], file_df["url"]):
        with open(folder_path+file_name, "wb") as f:
            relative_url = file_url.replace("https://amdcloud.sharepoint.com","")
            file = ctx.web.get_file_by_server_relative_url(relative_url).download(f).execute_query()
        print(f"File:'{file_name}' downloaded successfully")

def identify_filetype(filename:str):
    #Dynamically find the file types in a filename
    mypattern = r'\.[a-zA-Z0-9]+$'
    ext = re.findall(mypattern, filename)
    ext = ext[0].strip(".")
    return ext

def list_all_filetypes(filename_list):
    #Dynamically find the file types in the file objects
    exts_lst = []
    for file in filename_list:
        ext = identify_filetype(file)
        exts_lst.append(ext)
    filetypes = list(set(exts_lst))
    return filetypes

def create_file_df(file_objects):
    file_id_list = []
    file_name_list = []
    file_url_list = []
    file_modified_list = []
    for file in file_objects:
        id = file.unique_id
        name = file.name
        url = file.get_absolute_url().execute_query().value
        mod_dt = file.time_last_modified
        file_id_list.append(id)
        file_name_list.append(name)
        file_url_list.append(url)
        file_modified_list.append(mod_dt)
    lengths = [len(file_id_list),len(file_name_list), len(file_url_list), len(file_modified_list)]
    if lengths[0] == lengths[1] == lengths[2] == lengths[3]:
        file_df = pd.DataFrame(zip(file_id_list, file_name_list, file_url_list, file_modified_list))
        file_df.columns = ['file_id', 'name', 'url', 'last_modified']
        ext_list = []
        for name in list(file_df['name']):
            ext = identify_filetype(name)
            ext_list.append(ext)
        file_df['file_type'] = ext_list
        file_df['modified_dt'] = pd.to_datetime(file_df['last_modified'])
        return file_df
    else:
        print("Some file properties are missing.")

def get_file_df(sharepoint_url:str, parent_folder:str, username:str, password: str):
    ctx = create_sharepoint_client_context(sharepoint_url, username, password)
    myfiles = get_all_files_from_directory(ctx, parent_folder)
    df = create_file_df(myfiles)
    return df

def create_file_df_cloud(file_df, column_dict, load_dt):
    # convert the datetime column to date
    file_df[column_dict["column3"]] = pd.to_datetime(file_df[column_dict["column3"]])
    file_df[column_dict["column3"]] = file_df[column_dict["column3"]].dt.date

    # Get the list of documents that have changed after last load
    file_list_df = file_df[
        file_df[column_dict["column3"]] > datetime.strptime(load_dt, "%Y-%m-%d").date()
    ][[column_dict["column2"], column_dict["column1"], column_dict["column3"]]]
    file_list_df.reset_index(drop=True, inplace=True)

    # Derive seqno needed for file processing
    file_list_df[column_dict["column4"]] = file_list_df.index
    
    return file_list_df
    
def find_new_and_updated_files(file_objects, df, verbose=False):
    df['last_modified'] = pd.to_datetime(df['last_modified'])
    newfiles = []
    for file in file_objects:
        filename = file.name
        sp_timestamp = file.time_last_modified
        subdf = df[df['name']==filename].reset_index(drop=True)
        if subdf.shape[0] < 1:
            newfiles.append(file)
            if verbose:
                print("New file found: " + filename)            
        else:
            file_timestamp = subdf['last_modified'][0]
            if file_timestamp < sp_timestamp:
                newfiles.append(file) 
                if verbose:
                    print("Updated file found: " + filename) 
    return newfiles

def scrape_sharepoint(parent_url:str, relative_url:str, username:str, password: str, filepath:str, files_df=None, incremental=False, verbose=False):
    """ When performing an incremental update include your existing files_df, otherwise the entire directory will be pulled."""
    ctx = create_sharepoint_client_context(parent_url, username, password)
    all_files = get_all_files_from_directory(ctx, relative_url)
    if incremental:
        df = files_df
        newfiles = find_new_and_updated_files(all_files, df, verbose)
        download_files(newfiles, ctx, filepath, relative_url, verbose)
    else:
        download_files(all_files, ctx, filepath, relative_url, verbose)


#### On prem sharepoint ####
        
def create_onpremsharepoint_client_context(username:str, password:str):
    """Function for onprem sharepoint authentication.

    Args:
        username (str): onprem sharepoint username
        password (str): onprem sharepoint password

    Returns:
        session (object): authentication session
    """
    session = requests.Session()
    session.auth = HttpNtlmAuth(username, password)
    return session

def download_file_onprem(session, file_list_df, folder_path, chunk_size):
    """Function to download files from onprem sharepoint.

    Args:
        session (object): Shaepoint authentication session
        file_list_df (DataFrame): input dataframe
        folder_path (str): folder path to save downloaded files
        chunk_size (int): chunk size
    """
    for file_name, file_url in zip(
        file_list_df["name"], file_list_df["url"]
    ):
        response = session.get(file_url, stream=True,  headers = {"accept": "application/json;odata=verbose"})
        if response.status_code == 200:
            with open(folder_path + file_name, "wb") as f:
                for chunk in response.iter_content(int(chunk_size)):
                    if chunk:
                        f.write(chunk)

            print(f"File:'{file_name}' downloaded successfully")
        else:
            print(
                f"Failed to download the '{file_name}'. Response status code:",
                response.status_code,
            )
        
def create_file_df_onprem(file_url_list, session, column_dict, onprem_sharepoint_domain, load_dt):
    """Function to prepare dataframe for onprem source.

    Args:
        file_url_list (list): list of file urls
        session (object): authentication session for sharepoint
        column_dict (dict): dictionary of column names
        onprem_sharepoint_domain (str): sharepoint domain url
        load_dt (str): load date

    Returns:
        file_list_df (str): Dataframe with metadata extracted from onprem sharepoint files.
    """
    url_data = []
    
    for url in file_url_list:
        try:
            response = session.get(url, headers = {"accept": "application/json;odata=verbose"})
            responseJSON = response.json()

            match = re.search(r"decodedurl='(.*?)'", url)
            url_refined = match.group(1)

            resultsJSON = responseJSON["d"]["results"]

            for i in range(0, len(resultsJSON)):
                try:
                    url_data.append(
                        {
                            column_dict["column1"]: resultsJSON[i]["Name"],
                            column_dict["column2"]: url_refined,
                            column_dict["column3"]: resultsJSON[i]["TimeLastModified"],
                        }
                    )
                except Exception as e:
                    print("An exception has occured while deriving the attributes:", e)
        except Exception as e:
            print("An exception has occcured while connecting to sharepoint:", e)

    df_url = pd.DataFrame(url_data)
    df_url[column_dict["column3"]] = df_url[column_dict["column3"]].apply(
        lambda x: datetime.strptime(x.split("T")[0], "%Y-%m-%d").date()
    )
    # Get the list of documents that have changed after last load
    file_list_df = df_url[
        df_url[column_dict["column3"]] > datetime.strptime(load_dt, "%Y-%m-%d").date()
    ][[column_dict["column2"], column_dict["column1"], column_dict["column3"]]]
    file_list_df.reset_index(drop=True, inplace=True)

    # Derive the url and seqno needed for file processing
    file_list_df[column_dict["column4"]] = file_list_df.index
    file_list_df[column_dict["column2"]] = (
        onprem_sharepoint_domain + file_list_df[column_dict["column2"]] + "/" + file_list_df[column_dict["column1"]]
    )
    return file_list_df

### Weaviate data preparation ###

def prepare_weaviate_data_chunks(file_name, chunk_size, chunk_overlap_size):
    """Function to prepare parsed and cleaned text data chunks.

    Args:
        file_name (str): input file name for parsing
        chunk_size (int): chunk size
        chunk_overlap_size (int): chunk overlap size

    Returns:
        langchain_chunks (list): parsed and preprocessed text chunks
    """
    file_type = identify_filetype(file_name)

    if file_type == 'pdf':
        parsed_data = parsers.pdf_parse_into_pages(file_name)
    elif file_type == 'docx':
        parsed_data = parsers.doc_parser(file_name)
    elif file_type == 'pptx':
        parsed_data = parsers.pptx_parser(file_name)
    elif file_type=='txt':
        parsed_data = parsers.txt_parser(file_name)
    else:
        print(f"Parsing unavailable for {file_type}")
        
    formatted_text = parsers.format_text(parsed_data, chunk_size, chunk_overlap_size)
    langchain_chunks = [
        parsers.remove_unicode(str(doc.page_content)) for doc in formatted_text
    ]
    
    return langchain_chunks

def weaviate_vector_data_preparation(
    file_list_df,
    load_dt,
    column_dict, 
    folder_path,
    embedding_deployment_id,
    api_proxy_url,
    api_proxy_key,
    chunk_size,
    chunk_overlap_size,
    source="", 
    KB_article_flag = 'Y'
    
):
    """Function to prepare data in the form of dataframe.

    Args:
        file_list_df (DataFrame): Dataframe file name with list of files
        load_dt (str): timestamp in string format
        column_dict (dict): Dataframe columns passed in form of dictionary. 
        folder_path (str): Folder path of files
        embedding_deployment_id (str): LLM embeddings deployment id
        api_proxy_url (str): LLM API gateway proxy url
        api_proxy_key (str): LLM API gateway proxy key
        chunk_size (int): Chunk size
        chunk_overlap_size (int): chunk overlap size
        source (str): data source if present.  Defaults to "".
        KB_article_flag (str):  Y/N flag to check KB data is loading. Defaults to Y.

    Returns:
        DataFrame: Dataframe with generated embeddings.
    """
    df = pd.DataFrame(columns=list(column_dict.values()))
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # column1:filename, column2:url, column3:modified_dt
    for column1, column2, column3 in zip(
        file_list_df[column_dict["column1"]],
        file_list_df[column_dict["column2"]],
        file_list_df[column_dict["column3"]],
    ):
        chunks = prepare_weaviate_data_chunks(folder_path + column1, int(chunk_size), int(chunk_overlap_size))
        # column4:index, column5: text
        for column4, column5 in enumerate(chunks):
            # use cms function to get embeddings
            # column6: vector, column7:n_tokens, column8: chunk_id
            column6 = cmfunctions.get_embedding_llm(column5, embedding_deployment_id,api_proxy_url,
                                               api_proxy_key)
            column7 = len(tokenizer.encode(column5))
           
            column8 = str(uuid.uuid4())

            if KB_article_flag == 'Y':
                column5 = "Short Description : " + column1 + " Full Article: " + column5
            
            temp_df = pd.DataFrame(
                [
                    {
                        column_dict["column1"]: column1,
                        column_dict["column2"]: column2,
                        column_dict["column3"]: str(column3),
                        column_dict["column4"]: column4,
                        column_dict["column5"]: column5,
                        column_dict["column6"]: column6,
                        column_dict["column7"]: column7,
                        column_dict["column8"]: column8,
                        column_dict["column9"]: load_dt,
                        column_dict['column10']: source, 
                        column_dict['column11']: column1
                    }
                ]
            )
           

            df = pd.concat([df, temp_df])
            df = df.reset_index()
            df = df.drop(columns=['level_0'])

    return df