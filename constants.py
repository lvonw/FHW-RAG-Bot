import os
import enum
from dotenv import load_dotenv, find_dotenv
from enum import Enum
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dataclasses import dataclass


load_dotenv(find_dotenv())

# ========== PATHS ==========
PATH_VECTORDB = "./data/models/"
PATH_VECTORDB_SPLITTER = "./vectordb/splitter/"
PATH_VECTORDB_PDFLOADER = "./vectordb/pdfloader/"
PATH_PDF = "data/pdfs/"

# ========== LLM ==========
TEMPLATE = """ 
Beantworte die Frage ausschließlich mit dem hier gegebenen Kontext:
{context}

Frage: {question}"""

TEMPLATE_ALT = """ 
Beantworte die Frage ausschließlich mit dem hier gegebenen Kontext.
Solltest Du noch weitere Informationen benötigen erstelle bitte einen Prompt
für ein Sprachmodell welches diese Daten anfordert. Beginne diese Anfrage exakt
mit dem exakten Kennwort "PROMPT". Dieser prompt muss die originäre Frage 
enthalten:
{context}

Frage: {question}"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)
DEFAULT_GPT = "gpt-3.5-turbo-1106"
TEMPERATURE = 0.0
API_KEY = os.environ["OPENAI_API_KEY"]
LLM = {}


def setLLM(model_name):
    global LLM
    LLM = ChatOpenAI(
        model_name=model_name, temperature=TEMPERATURE, openai_api_key=API_KEY
    )

MAX_DOCUMENT_CHUNK_SIZE = 200 
SPLITTER = CharacterTextSplitter(chunk_size=MAX_DOCUMENT_CHUNK_SIZE, chunk_overlap=0, separator="(\\n\\n|\\.\\n)", is_separator_regex=True) #TODO Change Back
#SPLITTER = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

INPUT_PROMPT = "Wie kann ich Dir helfen?: "

AGENT_PROMPT = """
Verwende die Search-Funktion, um ähnliche Textabschnitte zu finden. Wenn der 
Textabschnitt nicht die gewünschte Antwort liefert, verwende die More-Funktion, 
um weitere Texte zu finden.
"""

SMART_AGENT_P = """
Verwende die Search-Funktion, um Textabschnitte zu finde, die die Frage beantworten könnten. 
Benutze die More funktion, um weitere Informationen zu finden.
Suche auch nach Synonymen, wenn die gesuchte Antwort nicht gefunden wurde.
Verwende zur beantwortung ausschließlich die, aus diesen Funktionen gewonnenen informationen. 
Bevor du antwortest, hohle einmal tief luft, entspanne dich und denke Schritt für Schritt. 
Gib bei deiner Antwort das Dokument und die Seite aus, auf welche du dich beziehst.
"""

SMART_AGENT_P_NM = """
Verwende die Search-Funktion, um Textabschnitte zu finde, die die Frage 
beantworten könnten. 
Verwende zur beantwortung ausschließlich die, aus diesen Funktionen gewonnenen 
informationen. 
Bevor du antwortest, hohle einmal tief luft, entspanne dich und
denke Schritt für Schritt. 
Gib bei deiner Antwort das Dokument und die Seite aus, auf welche
du dich beziehst.
"""

MAX_TOKENS = 1000


# ========== DATABASE ==========
class LoaderMethod(enum.Enum):
    CustomPDF_LOADER = "CustomLoader"
    PyPDF_LOADER = "PyPdfLoader"


DEFAULT_DATABASE = LoaderMethod.CustomPDF_LOADER


# ========== MODELS ==========
class ModelMethod(enum.Enum):
    VECSTORE = "VecStore"
    TOOLS = "TOOLS"
    CustomTool = "CustomTool"
    SMART_AGENT = "SmartAgent"
    OpenAI_ASSISTANT = "OpenAIAssistant"


# DEFAULT_MODEL = ModelMethod.TOOLS
# DEFAULT_MODEL = ModelMethod.CustomTool
# DEFAULT_MODEL = ModelMethod.VECSTORE
DEFAULT_MODEL = ModelMethod.SMART_AGENT


DEFAULT_DOC_AMOUNT = 32
USE_VERBOSE = True
MAX_ITERATIONS = 4

# ========== USAGE ==========
USAGE_PROGRAM_NAME = "FHDocsBot"
USAGE_PROGRAM_DESC = """
Beantwortet Fragen zu offiziellen Dokumenten der FH-Wedel.
"""
USAGE_ASK = """
Stellt die als Argument übergebene Frage an den Bot.
"""
USAGE_DATABASE = """
Spezifiziert die zu nutzende Vektor-Datenbank.
"""
USAGE_MODEL = """
Spezifiziert das zu nutzende Model.
"""
USAGE_GPT = """
Gibt die zu verwendene openAi-Gpt Version an.
"""
USAGE_INIT = """
Die Vektor-Datenbank wird neu erstellt.
"""
USAGE_VALIDATE = """
Es werden Testfragen ausgegeben.
"""
USAGE_CLOSEST_V = """
Gibt die aus der Vektor-Datenbank gelesenen Dokumente aus, anstelle der Antwort.
"""
USAGE_CLI = """
Es wird die Kommandozeile verwendet, um eine Frage zu stellen.
"""


# ========== REST ==========
@dataclass
class DefaultArgs:
    question: str = ""
    database: str = DEFAULT_DATABASE
    model: str = DEFAULT_MODEL
    init: bool = False
    validate: bool = False
    cv: bool = False
    cli: bool = False
    gpt: str = DEFAULT_GPT
