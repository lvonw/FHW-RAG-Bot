import os
import argparse
import constants

from pdfloader                      import PDFCustomLoader

from langchain.schema               import StrOutputParser
from langchain.schema.runnable      import RunnablePassthrough
from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.text_splitter        import CharacterTextSplitter
from langchain.vectorstores         import Chroma


def get_user_input():
    return input(constants.INPUT_PROMPT)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def ask_question(question, retriever):
    question_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | constants.PROMPT 
        | constants.LLM 
        | StrOutputParser())
    return question_chain.invoke(question)
        
def get_retriever(mode=constants.DEFAULT_DATABASE, init=False):
    db = None
    path = get_path(mode)
    if init:
        docs = []
        match mode:
            case constants.TokenizeMethod.CHAR_SPLITTER: 
                loader      = PyPDFDirectoryLoader(path=constants.PATH_PDF) 
                splitter    = CharacterTextSplitter(chunk_size=1000, 
                                                    chunk_overlap=0)
                docs        = loader.load_and_split(text_splitter=splitter)
            case constants.TokenizeMethod.PDF_LOADER: 
                files = os.listdir(constants.PATH_PDF)
                for file in files:
                    loader = PDFCustomLoader(file_path=constants.PATH_PDF+file)
                    docs += loader.load()
        db = init_and_persist_chroma(docs, path)
    else:
        db = load_chroma(path)
    
    return db.as_retriever()

def init_and_persist_chroma(docs, path): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory = path)
    return db

def load_chroma(path):
    return Chroma(
        embedding_function = OpenAIEmbeddings(openai_api_key=constants.API_KEY),
        persist_directory  = path) 
        
def get_path(mode):
    match mode:
        case constants.TokenizeMethod.CHAR_SPLITTER: 
            return constants.PATH_VECTORDB_SPLITTER        
        case constants.TokenizeMethod.PDF_LOADER: 
            return constants.PATH_VECTORDB_PDFLOADER

def get_db_choices():
    return[m.value for m in constants.TokenizeMethod]

def parse_db_choice(choice):
    try:
        return constants.TokenizeMethod[choice]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid database option")
    
def prepare_arg_parser():
    parser = argparse.ArgumentParser(prog=constants.USAGE_PROGRAM_NAME,
                                     description=constants.USAGE_PROGRAM_DESC)
    parser.add_argument("-a",
                        "--ask",
                        dest="question", 
                        nargs=1,
                        help=constants.USAGE_ASK,
                        type=str)
    parser.add_argument("-db",
                        "--database",
                        dest="database",
                        choices=get_db_choices(),
                        default=constants.DEFAULT_DATABASE,
                        nargs=1,
                        help=constants.USAGE_DATABASE,
                        type=parse_db_choice) 
    parser.add_argument("-i", 
                        "--init", 
                        dest="init",
                        action="store_true",
                        help=constants.USAGE_INIT)
    parser.add_argument("-cv",
                        "--closest-vectors",
                        dest="cv",
                        action="store_true",
                        help=constants.USAGE_CLOSEST_V)
    parser.add_argument("-c",
                        "--cli",
                        dest="cli",
                        action="store_true",
                        help=constants.USAGE_CLI)
    return parser

def main():
    parser = prepare_arg_parser()
    args = parser.parse_args()    
    # Get the correct database and initialize it if demanded
    retriever = get_retriever(args.database, args.init)    
    # Handle the question answering
    if args.question or args.cli:
        # Get the question 
        user_input = None
        if args.question:
            user_input = args.question[0]
        elif args.cli:
            user_input = get_user_input()
        # Answer the question or show the relevant vectors
        if args.cv:
            print(retriever.invoke(user_input))
        else:
            print(ask_question(user_input, retriever))

if __name__ == "__main__":
    main()