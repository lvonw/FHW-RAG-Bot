import os
import argparse
import constants
from pdfloader import PDFCustomLoader
from enum import Enum

from langchain.schema               import StrOutputParser
from langchain.schema.runnable      import RunnablePassthrough
from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.text_splitter        import CharacterTextSplitter
from langchain.vectorstores         import Chroma

class TokenizeMethod(Enum):
    PDF_LOADER = 0
    CHAR_SPLITTER = 1

INIT_CHROMA = False
MODE = TokenizeMethod.PDF_LOADER

# ============ MAIN ============
def ask_question(question, retriever):
    question_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | constants.PROMPT 
        | constants.LLM 
        | StrOutputParser())
    return question_chain.invoke(question)
    
def init_and_persist_chroma(docs, path): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory = path)
    return db

def load_chroma(path):
    return Chroma(
        embedding_function  = OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory   = path) 
        
def get_user_input():
    return input(constants.INPUT_PROMPT)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def main():
    db = None
    path = constants.PATH_VECTORDB_PDFLOADER if MODE == TokenizeMethod.PDF_LOADER else constants.PATH_VECTORDB_SPLITTER
    if INIT_CHROMA:
        docs = []
        match MODE:
            case TokenizeMethod.CHAR_SPLITTER: 
                loader      = PyPDFDirectoryLoader(path=constants.PATH_PDF) 
                splitter    = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs        = loader.load_and_split(text_splitter=splitter)
            case  TokenizeMethod.PDF_LOADER: 
                files = os.listdir(constants.PATH_PDF)
                # Print each filename
                for file in files:
                    loader = PDFCustomLoader(file_path= constants.PATH_PDF + file)
                    docs += loader.load()
        db = init_and_persist_chroma(docs, path)
    else:
        db = load_chroma(path)
    
    retriever = db.as_retriever()

    user_input = get_user_input()

    print(ask_question(user_input, retriever))
    print("===================")



if __name__ == "__main__":
    main()