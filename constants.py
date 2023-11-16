import os
from dotenv                         import load_dotenv, find_dotenv
from enum                           import Enum
from langchain.chat_models          import ChatOpenAI
from langchain.prompts              import PromptTemplate

load_dotenv(find_dotenv())

# ========== PATHS ==========
PATH_VECTORDB_SPLITTER      = "./data/vectordb/splitter/"
PATH_VECTORDB_PDFLOADER     = "./data/vectordb/pdfloader/"
PATH_PDF        = "data/pdfs/"

# ========== LLM ==========
TEMPLATE        = """ 
Beantworte die Frage ausschließlich mit dem hier gegebenen Kontext:
{context}

Frage: {question}""" 
PROMPT = PromptTemplate(
    input_variables = ["context", "question"],
    template        = TEMPLATE)
GPT_TURBO       = "gpt-3.5-turbo"
TEMPERATURE     = 0.0
API_KEY         = os.environ["OPENAI_API_KEY"]
LLM             = ChatOpenAI(  
    model_name      = GPT_TURBO, 
    temperature     = TEMPERATURE, 
    openai_api_key  = API_KEY)

INPUT_PROMPT    = "Wie kann ich Dir helfen?: "

# ========== DATABASE ==========
class TokenizeMethod(Enum):
    PDF_LOADER = "PDF Loader"
    CHAR_SPLITTER = "Character Splitter"

INIT_CHROMA = False
DEFAULT_DATABASE = TokenizeMethod.PDF_LOADER

# ========== USAGE ==========
USAGE_PROGRAM_NAME  = """
FHDocsBot
"""  
USAGE_PROGRAM_DESC  = """
Beantwortet Fragen zu offiziellen Dokumenten der FH-Wedel.
""" 
USAGE_ASK           = """
Stellt die als Argument übergebene Frage an den Bot.
"""
USAGE_DATABASE      = """
Spezifiziert die zu nutzende Vektor-Datenbank.
"""
USAGE_INIT          = """
Die Vektor-Datenbank wird neu erstellt.
"""
USAGE_CLOSEST_V     = """
Gibt die aus der Vektor-Datenbank gelesenen Dokumente aus, anstelle der Antwort.
"""
USAGE_CLI           = """
Es wird die Kommandozeile verwendet, um eine Frage zu stellen.
"""