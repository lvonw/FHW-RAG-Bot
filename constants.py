import os
from dotenv                         import load_dotenv, find_dotenv
from enum                           import Enum
from langchain.chat_models          import ChatOpenAI
from langchain.prompts              import PromptTemplate

load_dotenv(find_dotenv())

PATH_VECTORDB_SPLITTER      = "./data/vectordb/splitter/"
PATH_VECTORDB_PDFLOADER     = "./data/vectordb/pdfloader/"
PATH_PDF        = "data/pdfs/"

INPUT_PROMPT    = "Wie kann ich Dir helfen?: "

GPT_TURBO       = "gpt-3.5-turbo"
TEMPERATURE     = 0.0
API_KEY         = os.environ["OPENAI_API_KEY"]
LLM             = ChatOpenAI(  
    model_name      = GPT_TURBO, 
    temperature     = TEMPERATURE, 
    openai_api_key  = API_KEY)

TEMPLATE        = """ 
Beantworte die Frage ausschließlich mit dem hier gegebenen Kontext:
{context}

Frage: {question}""" 

PROMPT = PromptTemplate(
    input_variables = ["context", "question"],
    template        = TEMPLATE)

class TokenizeMethod(Enum):
    PDF_LOADER = "PDF Loader"
    CHAR_SPLITTER = "Character Splitter"

INIT_CHROMA = False
DEFAULT_DATABASE = TokenizeMethod.PDF_LOADER
