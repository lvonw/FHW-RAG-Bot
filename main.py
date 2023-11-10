import os

from dotenv                         import load_dotenv, find_dotenv

from langchain.chat_models          import ChatOpenAI
from langchain.prompts              import PromptTemplate
from langchain.schema               import StrOutputParser

from langchain.document_loaders     import PyPDFDirectoryLoader
from langchain.embeddings.openai    import OpenAIEmbeddings
from langchain.text_splitter        import CharacterTextSplitter
from langchain.vectorstores         import Chroma
from langchain.schema.runnable      import RunnablePassthrough


# Load the environment variables. 
load_dotenv(find_dotenv())


# Constants
GPT_TURBO   = "gpt-3.5-turbo"
TEMPERATURE = 0.0
API_KEY     = os.environ["OPENAI_API_KEY"]
INIT_CHROMA = False

LLM         = ChatOpenAI(  
            model_name      = GPT_TURBO, 
            temperature     = TEMPERATURE, 
            openai_api_key  = API_KEY)

def init_and_persist_chroma(docs): 
    db = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(openai_api_key=API_KEY),
        persist_directory   = "./data/vectordb/")
    return db


def load_chroma():
    return Chroma(
        embedding_function  = OpenAIEmbeddings(openai_api_key=API_KEY),
        persist_directory   = "./data/vectordb/") 
        

# ============ MAIN ============
def get_user_input():
    return input("Please enter the fact you want to get verified: ")

template = """ 
Beantworte die Frage ausschließlich mit dem hier gegebenen Kontext:
{context}

Frage: {question}
""" 
# Prompts
prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template        = template
)




def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
    #return docs[0].page_content

def main():
    db = None
    if INIT_CHROMA:
        loader      = PyPDFDirectoryLoader(path="data/pdfs/") 
        splitter    = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs        = loader.load_and_split(text_splitter=splitter)
        db          = init_and_persist_chroma(docs)
    else:
        db = load_chroma()
    
    retriever = db.as_retriever()
    question1 = "Was kannst du mir zum Thema Freiwilliges auslandssemster sagen"
    question2 = "Welche modul nr hat Grundlagen der Computergrafik?"

    #print(retriever.invoke(question)[0].page_content)
    chain1 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | LLM 
        | StrOutputParser())

    # result = chain1.invoke(question1)
    # print(result)
    # print("===================")
    # result = chain1.invoke(question2)
    # print(result)
    # print("===================")

    print(format_docs(retriever.invoke(question2)))
    #print(format_docs(retriever.invoke(question1)))
    # print(format_docs(retriever.invoke(question1)))
    result = chain1.invoke(question2)
    print(result)
    print("===================")


    # print(len(retrieved_docs))
    # print(retrieved_docs[0].page_content)



if __name__ == "__main__":
    main()