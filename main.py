import os

from dotenv                         import load_dotenv, find_dotenv

from langchain.chat_models          import ChatOpenAI
from langchain.prompts              import PromptTemplate
from langchain.schema               import StrOutputParser

from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.text_splitter        import CharacterTextSplitter
from langchain.vectorstores         import Chroma

# Load the environment variables. 
load_dotenv(find_dotenv())




# docs = loader.load()

# Constants
GPT_TURBO   = "gpt-3.5-turbo"
TEMPERATURE = 0.3
API_KEY     = os.environ["OPENAI_API_KEY"]
INIT_CHROMA = False

# LLM         = ChatOpenAI(  
#            model_name      = GPT_TURBO, 
#            temperature     = TEMPERATURE, 
#            openai_api_key  = API_KEY)

def init_and_persist_chroma(docs): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=API_KEY),
        persist_directory   = "./data/vectordb/")
    return db


def load_chroma():
    return Chroma(
        embedding_function  = OpenAIEmbeddings(openai_api_key=API_KEY),
        persist_directory   = "./chroma_db") 
        

# ============ MAIN ============
def get_user_input():
    return input("Please enter the fact you want to get verified: ")



def main():
    db = None
    if INIT_CHROMA:
        loader      = PyPDFDirectoryLoader(path="data/pdfs/") 
        splitter    = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs        = loader.load_and_split(text_splitter=splitter)
        db          = init_and_persist_chroma(docs)
    else:
        db = load_chroma()
    # print(db)

if __name__ == "__main__":
    main()