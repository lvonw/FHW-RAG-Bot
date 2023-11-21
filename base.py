import os
from typing import Callable, List

import constants

from pdfloader                      import PDFCustomLoader

from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.vectorstores.chroma  import Chroma

from abc import ABC, abstractmethod
from langchain.document_loaders.base import Document

from langchain.schema.runnable.base import Runnable

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

class ModelBase(ABC):
    @abstractmethod
    def init(self, mode : constants.TokenizeMethod, init : bool):
        pass

    @abstractmethod
    def getModel(self):
        pass


def init_and_persist_chroma(docs, path): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory = path)
    return db

def load_chroma(path):
    return Chroma(
        embedding_function = OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory  = path) 

def get_loader(mode, glob = "**/*.pdf"):
    match mode:
        case constants.TokenizeMethod.CHAR_SPLITTER: 
            return PyPDFDirectoryLoader(path=constants.PATH_PDF) 
        case constants.TokenizeMethod.PDF_LOADER: 
            return DirectoryLoader(constants.PATH_PDF, glob=glob, loader_cls=PDFCustomLoader)


def get_path(mode, name):
    match mode:
        case constants.TokenizeMethod.CHAR_SPLITTER: 
            return os.path.join(constants.PATH_VECTORDB, name, constants.PATH_VECTORDB_SPLITTER)       
        case constants.TokenizeMethod.PDF_LOADER: 
            return os.path.join(constants.PATH_VECTORDB, name, constants.PATH_VECTORDB_PDFLOADER)

def get_retriever(name, init_func  : Callable[[], List[Document] ], mode=constants.DEFAULT_DATABASE, init=False):
    db = None
    path = get_path(mode, name)
    if init:
        docs = init_func(mode)
        db = init_and_persist_chroma(docs, path)
    else:
        db = load_chroma(path)
    
    return db.as_retriever()

def get_retrieverTool(name, init_func  : Callable[[], List[Document] ], mode=constants.DEFAULT_DATABASE, init=False, description=""):
    docstore = RetrievalQA.from_chain_type(constants.LLM, retriever=get_retriever(name, init_func, mode, init))
    return Tool(
            name=name,
            func=docstore.run,
            description=description,
        )