
from io import TextIOWrapper
from base import ModelBase, get_loader, get_retriever, write_validation_retriever

import constants

from langchain.schema               import StrOutputParser
from langchain.schema.runnable      import RunnablePassthrough

def getRetriever(mode, init):
    def loader1(mode):
        return get_loader(mode, "**/*.pdf").load_and_split(constants.SPLITTER)
    return get_retriever("retriever", loader1, mode, init)

def format_docs(validation_file : TextIOWrapper | None):
    def format(docs):
        doc =  "\n\n".join([d.page_content for d in docs])
        write_validation_retriever(validation_file, doc)
        return doc
    return format

def getModel(retriever, validation_file : TextIOWrapper | None):
    question_chain = (
        {"context": retriever | format_docs(validation_file), "question": RunnablePassthrough()} 
        | constants.PROMPT 
        | constants.LLM 
        | StrOutputParser())
    return question_chain


class Model(ModelBase):
    def init(self, mode, init):
        self.retriever = getRetriever(mode, init)

    def getModel(self, validation_file : TextIOWrapper | None):
        return getModel(self.retriever,validation_file)
        
    