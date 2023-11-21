
from base import ModelBase, get_loader, get_retriever

import constants

from langchain.schema               import StrOutputParser
from langchain.schema.runnable      import RunnablePassthrough

def getRetriever(mode, init):
    def loader1(mode):
        return get_loader(mode, "**/*.pdf").load_and_split(constants.SPLITTER)
    return get_retriever("Retriever", loader1, mode, init)

def format_docs(docs):
    doc =  "\n\n".join([d.page_content for d in docs])
    print(doc)
    return doc

def getModel(retriever):
    question_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | constants.PROMPT 
        | constants.LLM 
        | StrOutputParser())
    return question_chain


class Model(ModelBase):
    def init(self, mode, init):
        self.retriever = getRetriever(mode, init)

    def getModel(self):
        return getModel(self.retriever)
        
    