from setuptools import setup, find_packages
DESCRIPTION = "Python package for Data Ingestion"
LONG_DESCRIPTION = "This Python package is used to perform text cleaning, scraping data from Sharepoint, \
                    accessing and storing vectors in Weaviate vectorDB,helps to eaily access the LLM Gateway."

setup(
    name='llmcore',
    version='0.1.7',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=['Office365-REST-Python-Client', 'tiktoken', 'weaviate-client','langchain','openai','pandas','openpyxl','tabula','extract-msg','unstructured[all-docs]', 'requests-ntlm']
)