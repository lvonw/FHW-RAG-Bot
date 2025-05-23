from io import TextIOWrapper
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
import shutil

class ModelBase(ABC):
    @abstractmethod
    def init(self, mode : constants.LoaderMethod, init : bool):
        pass

    @abstractmethod
    def getModel(self, validation_file : TextIOWrapper | None):
        pass
class colors:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
    
    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'
    
    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'

class Logger:
    def __init__(self) -> None:
        self.output = ""

    def set_validation_file(self,validation_file):
        self.validation_file = validation_file
    def log(self, text, color, color_validation) -> 'Logger':
        self.output += text + "\n"
        print(color + text)
        if hasattr(self, "validation_file"):
            write_validation_color(self.validation_file, text, color_validation)
        return self
    def log_red(self, text)  -> 'Logger':
        return self.log(text, colors.fg.red, "red")
    def log_green(self, text)  -> 'Logger':
        return self.log(text, colors.fg.green, "green")
    def log_black(self, text)  -> 'Logger':
        return self.log(text, colors.fg.black, "black")
    def log_blue(self, text)  -> 'Logger':
        return self.log(text, colors.fg.blue, "blue")
            


#from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
#from langchain.embeddings import AlephAlphaSymmetricSemanticEmbedding
#from langchain.embeddings import TensorflowHubEmbeddings

def getEmbedding():
    embeddings = OpenAIEmbeddings(openai_api_key=constants.API_KEY)
    return embeddings
#embeddings = AlephAlphaAsymmetricSemanticEmbedding(normalize=True, compress_to_size=128)
#embeddings = AlephAlphaSymmetricSemanticEmbedding(normalize=True, compress_to_size=128)
#embeddings = TensorflowHubEmbeddings()


def init_and_persist_chroma(docs, path): 
    db = Chroma.from_documents(docs, getEmbedding(),persist_directory = path)
    return db

def load_chroma(path):
    return Chroma(
        embedding_function = getEmbedding(),
        persist_directory  = path) 

def get_loader(mode, glob = "**/*.pdf"):
    match mode:
        case constants.LoaderMethod.PyPDF_LOADER: 
            return PyPDFDirectoryLoader(path=constants.PATH_PDF) 
        case constants.LoaderMethod.CustomPDF_LOADER: 
            return DirectoryLoader(constants.PATH_PDF, glob=glob, loader_cls=PDFCustomLoader, show_progress=True)


def get_path(mode, name):
    match mode:
        case constants.LoaderMethod.PyPDF_LOADER: 
            return os.path.join(constants.PATH_VECTORDB, name, constants.PATH_VECTORDB_SPLITTER)       
        case constants.LoaderMethod.CustomPDF_LOADER: 
            return os.path.join(constants.PATH_VECTORDB, name, constants.PATH_VECTORDB_PDFLOADER)

def get_retriever(name, init_func  : Callable[[], List[Document] ], mode=constants.DEFAULT_DATABASE, init=False, k = 1):
    db = None
    path = get_path(mode, name)

    if not os.path.isdir(path):
        init = True
    if init:
        if os.path.isdir(path):
            shutil.rmtree(path)
        docs = init_func(mode)
        db = init_and_persist_chroma(docs, path)
    else:
        if os.path.exists(path):
            db = load_chroma(path)
        else:
            print ("ERROR: Database has not been initialized")
            return None
    return db.as_retriever(search_kwargs = {'k': k})
    #return db.as_retriever(search_type='mmr')

def get_retrieverTool(name, init_func  : Callable[[], List[Document] ], mode=constants.DEFAULT_DATABASE, init=False, description=""):
    docstore = RetrievalQA.from_chain_type(constants.LLM, retriever=get_retriever(name, init_func, mode, init))
    return Tool(
            name=name,
            func=docstore.run,
            description=description
            )

def write_validation_color(file, text, color):
    if file is not None:
        file.write(f"<font color=\"{color}\">{text}</font>\n")

def write_validation_retriever(file, text):
    write_validation_color(file, text, "red")