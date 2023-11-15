import os
import argparse
import constants

from langchain.schema               import StrOutputParser
from langchain.schema.runnable      import RunnablePassthrough
from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.text_splitter        import CharacterTextSplitter
from langchain.vectorstores         import Chroma

INIT_CHROMA = False

# ============ MAIN ============
def ask_question(question, retriever):
    question_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | constants.PROMPT 
        | constants.LLM 
        | StrOutputParser())
    return question_chain.invoke(question)
    
def init_and_persist_chroma(docs): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory   = constants.PATH_VECTORDB)
    return db

def load_chroma():
    return Chroma(
        embedding_function  = OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory   = constants.PATH_VECTORDB) 
        
def get_user_input():
    return input(constants.INPUT_PROMPT)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def main():
    db = None
    if INIT_CHROMA:
        loader      = PyPDFDirectoryLoader(path=constants.PATH_PDF) 
        splitter    = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs        = loader.load_and_split(text_splitter=splitter)
        db          = init_and_persist_chroma(docs)
    else:
        db = load_chroma()
    
    retriever = db.as_retriever()

    user_input = get_user_input()


    print(ask_question(user_input, retriever))
    print("===================")


if __name__ == "__main__":
    main()