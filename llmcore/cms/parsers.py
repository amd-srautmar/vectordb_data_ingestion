import pandas as pd  # NOT USED
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,  
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader,
    PyPDFLoader
)
from langchain.docstore.document import Document
import tabula
import re
import extract_msg
import tiktoken


# ------------------------------------------------------------------------
# Weaviate fucntions in
# https://weaviate.io/developers/academy/standalone/chunking/how_1
# -------------------------------------------------------------------------


def word_splitter(source_text: str) -> List[str]:
    """
    source_text: text to be split

    >>> text = ('I would like to thank my middle finger for always sticking'
    ...         ' up               for me when I       needed it.')
    >>> chunks = word_splitter(text)
    >>> chunks
    ['I', 'would', 'like', 'to', 'thank', 'my', 'middle', 'finger', 'for', 'always', 'sticking', 'up', 'for', 'me', 'when', 'I', 'needed', 'it.']
    """
    source_text = re.sub(r"\s+", " ", source_text)  # Replace multiple whitespces

    return re.split(r"\s", source_text)  # Split by single whitespace


def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    """
    text: source text to chunk
    chunk_size: token size of each chunk
    overlap_fraction: percentange of chunksize to overlap between chunks

    >>> text = ('I would like to thank my middle finger for always sticking'
    ...         ' up for me when I needed it.')
    >>> chunks = get_chunks_fixed_size_with_overlap(text, 5, 0.2)
    >>> chunks
    ['I would like to thank', 'thank my middle finger for always', 'always sticking up for me when', 'when I needed it.']

    # Below is demonstrating how to create the same langchain structure
    # metadata could be stored in a dataframe column and the chunk in
    # an other column 
    >>> metadata = {'url': 'http://theurl.com', 'topic': 'servival techniques'}
    >>> embedding_chunks = [(chunk, metadata) for chunk in chunks]
    >>> embedding_chunks
    [('I would like to thank', {'url': 'http://theurl.com', 'topic': 'servival techniques'}), ('thank my middle finger for always', {'url': 'http://theurl.com', 'topic': 'servival techniques'}), ('always sticking up for me when', {'url': 'http://theurl.com', 'topic': 'servival techniques'}), ('when I needed it.', {'url': 'http://theurl.com', 'topic': 'servival techniques'})]
    """
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks

# -----------------------------------------------------------------------
# Tested Parsers
# -------------------------------------------------------------------------


def pptx_parser(pptx_file):
    loader = UnstructuredPowerPointLoader(pptx_file)
    data = loader.load()

    return data


def pdf_parse_into_pages(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()

    return pages


def doc_parser(word_file):
    loader = UnstructuredWordDocumentLoader(word_file)
    data = loader.load()

    return data


def msg_parser(filepath):
    msg = extract_msg.openMsg(filepath)
    msg_json = msg.getJson()

    return msg_json


def save_email_attachments(msgdf, path):
    for filename in msgdf['file_name']:
        filepath = path+'/'+filename
        msg = extract_msg.openMsg(filepath)
        for att in msg.attachments:
            print("Saving " + str(att.longFilename))
            att.save(customPath=path)

def txt_parser(txt_file):
    loader =  UnstructuredFileLoader(txt_file)
    data = loader.load()
    return data
# -------------------------------------------------------------------------
# UNTested Parsers
# -------------------------------------------------------------------------


def email_parser(filepath):
    loader = UnstructuredEmailLoader(filepath)
    data = loader.load()

    return data


def process_pdf_table(pdf_bytes):
    with open("tmp.pdf", "wb") as file:
        file.write(pdf_bytes.read())

    tables = tabula.read_pdf("tmp.pdf", pages="all", encoding='latin1')
    docs = []
    for table in tables:
        table_string = table.to_csv(sep='|', index=False, header=True)
        docs.append(Document(page_content=table_string, metadata={"source": "pdf"}))

    return docs


def pdf_parser(pdf_file):
    loader = UnstructuredPDFLoader(pdf_file)
    data = loader.load()

    return data


# TODO: Is this doing a similar thing as get_chunks_fixed_size_with_overlap above, but with metadata?
#       - also may be better to read in files outside a function
def text_parser(text_file):
    content = text_file.read().decode("utf-8")
    metadata = {"source": text_file}
    data = [Document(page_content=content, metadata=metadata)]

    return data

# -------------------------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------------------------

# TODO: Is this doing a similar thing as get_chunks_fixed_size_with_overlap above, but with metadata?
def format_text(data, chunk_size, chunk_overlap_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap_size)
    texts = text_splitter.split_documents(data)
    #df_pdf = pd.DataFrame([t.page_content for t in texts], columns =['text'])

    return texts


# TODO: Is this doing a similar thing as text_splitter above?
def remove_unicode(string):
    new_string = string.replace("\n", "").replace("\r", "")
    strencode = new_string.encode("ascii", "ignore")
    cleanstr = strencode.decode()
    cleanstr = (
        cleanstr.replace("\\r", "")
        .replace("\\n", "")
        .replace("\\t", "")
        .replace("\\", "")
    )
    cleanstr = re.sub(r"\s+", " ", cleanstr)  # Replace multiple whitespces

    return cleanstr


# TODO: Is this doing a similar thing as text_splitter above?
def clean_documents(documents):
    cleaned_documents = []
    for doc in documents:
        mystr = str(doc.page_content)
        cleanstr = re.sub(r"\s+", " ", mystr)  # Replace multiple whitespces
        cleaned_doc = remove_unicode(cleanstr)
        cleaned_documents.append(cleaned_doc)

    return cleaned_documents


def add_text_to_docdf(docdf, path):
    txtlst = []
    for doc in docdf['file_name']:
        myfile = doc
        myfilepath = os.path.join(path, myfile)
        data = doc_parser(myfilepath)
        cleandata = remove_unicode(data[0].page_content)
        txtlst.append(cleandata)
    docdf['txt'] = txtlst
    tokenizer = tiktoken.get_encoding("cl100k_base")
    docdf['n_tokens'] = docdf['txt'].apply(lambda x: len(tokenizer.encode(x)))

    return docdf


def add_text_to_pdfdf(pdfdf, path):
    txtlst = []
    for pdf in pdfdf['file_name']:
        myfile = pdf
        myfilepath = path+'/'+myfile
        pages = pdf_parse_into_pages(myfilepath)
        pg_list = []
        for page in pages:
            cleandata = remove_unicode(page.page_content)
            pg_list.append(cleandata)
        s = ' '
        pg_list = s.join(pg_list)
        txtlst.append(pg_list)
    pdfdf['txt'] = txtlst
    tokenizer = tiktoken.get_encoding("cl100k_base")
    pdfdf['n_tokens'] = pdfdf['txt'].apply(lambda x: len(tokenizer.encode(x)))

    return pdfdf


def add_text_to_pptdf(pptdf, path):
    txtlst = []
    for ppt in pptdf['file_name']:
        myfile = ppt
        myfilepath = path+'/'+myfile
        data = pptx_parser(myfilepath)
        cleandata = remove_unicode(data[0].page_content)
        txtlst.append(cleandata)
    pptdf['txt'] = txtlst
    tokenizer = tiktoken.get_encoding("cl100k_base")
    pptdf['n_tokens'] = pptdf['txt'].apply(lambda x: len(tokenizer.encode(x)))

    return pptdf


def add_clean_text_to_df(df, filetype, path):
    supported_filetypes = ['pptx', 'pdf', 'docx']
    if filetype not in supported_filetypes:
        error_msg = "Error.  Unsupported filetype given. Only these filetypes are supported:"
        print(error_msg)
        print(supported_filetypes)
    else:
        if filetype == 'pdf':
            newdf = add_text_to_pdfdf(df, path)

        elif filetype == 'pptx':
            newdf = add_text_to_pptdf(df, path)

        elif filetype == 'docx':
            newdf = add_text_to_docdf(df, path)

        return newdf
