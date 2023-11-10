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


loader = PyPDFDirectoryLoader(path="data/pdfs/") 
# docs = loader.load_and_split
docs = loader.load()

db = Chroma.from_documents(docs)
# Constants
# GPT_TURBO   = "gpt-3.5-turbo"
# TEMPERATURE = 0.3
# API_KEY     = os.environ["OPENAI_API_KEY"]
# LLM         = ChatOpenAI(
#            model_name      = GPT_TURBO, 
#            temperature     = TEMPERATURE, 
#            openai_api_key  = API_KEY)


# ============ MAIN ============
def get_user_input():
    return input("Please enter the fact you want to get verified: ")

def main():
    print(len(docs))
    for d in docs:
        print (d.metadata)
    print(db)

if __name__ == "__main__":
    main()