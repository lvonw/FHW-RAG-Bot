from io import TextIOWrapper
import constants
import os

from base                               import (ModelBase, 
                                                get_loader, 
                                                get_retriever, write_validation_color, write_validation_retriever)
from langchain.agents.agent             import AgentExecutor
from langchain.schema.vectorstore       import  VectorStoreRetriever
from langchain.schema.runnable          import RunnableLambda
from langchain.prompts                  import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render             import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers    import OpenAIFunctionsAgentOutputParser
from langchain.agents                   import AgentExecutor
from langchain.tools                    import tool

from typing                             import List
from langchain.schema.document          import Document


def get_custom_retriever(retriever : VectorStoreRetriever,validation_file : TextIOWrapper | None ):
    docs = None 
    idx = 0
    counter = 0
    @tool
    def search(query : str) -> str:
        """Gets documents with similar content

        Args:
            text: text to search for
        """
        nonlocal docs
        nonlocal idx
        nonlocal counter
        idx = 0
        counter += 1
        write_validation_color(validation_file, "Query:" + query, "")

        if counter > constants.MAX_ITERATIONS:
            write_validation_color(validation_file, "End of Query", "")
            return "Es gibt keine weiteren Daten. Gib jetzt das beste Ergebnis aus, das du hast."

        # Init the documents
        docs = retriever_call(retriever, query, constants.DEFAULT_DOC_AMOUNT)
        res, idx = get_dynamic_doc_amount(docs, idx, validation_file)
        return res

    @tool
    def more() -> str:
        """Returns more documents of similar content already searched, only
        use if search didnt return enough information to answer the quesiton
        sufficiently 
        """
        nonlocal docs
        nonlocal idx
        nonlocal counter
        counter += 1

        if counter > constants.MAX_ITERATIONS:
            write_validation_color(validation_file, "End of Query", "")
            return "Es gibt keine weiteren Daten. Gib jetzt das beste Ergebnis aus, das du hast."

        res, idx = get_dynamic_doc_amount(docs, idx, validation_file)
        return res
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",  constants.SMART_AGENT_P),
        ("human",   "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])

    tools           = [more, search]
    #tools           = [search]
    tool_funs       = [format_tool_to_openai_function(t) for t in tools]
    llm_with_tools  = constants.LLM.bind(functions = tool_funs)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent = AgentExecutor(agent=agent, tools=tools, verbose=constants.USE_VERBOSE)
    return RunnableLambda(lambda x: {"input" : x}) | agent

def retriever_call(r : VectorStoreRetriever, q, k) -> List[Document]:
    return r.get_relevant_documents(q, search_kwargs = {'k': k})

def compute_token_amount(doc : str):
    return (len(doc) / 4) 

def get_docs_content(doc : Document) -> str:
    metadata = get_formatted_metadata(doc)
    content = doc.page_content
    return metadata + content

def get_formatted_metadata(doc : Document) -> str:
    res = ""
    metadata = doc.metadata
    source = metadata["source"]

    filename = os.path.basename(source)
    title, _ = os.path.splitext(filename)

    res += f"In dem Dokument {title} befinden sich"
    
    if "page" in metadata:
        page = metadata["page"]
        res += f", auf der Seite {page},"

    res += " die folgenden Informationen: \n"
    return res


def get_dynamic_doc_amount(docs : List[Document], startIdx, validation_file) -> str:
    length = 0 
    idx = startIdx
    result = ""

    while ((idx < len(docs))
            and ((length + compute_token_amount(get_docs_content(docs[idx])) < constants.MAX_TOKENS) or not length)):   
        
        result += ("" if result == "" else "\n\n") + get_docs_content(docs[idx])
        length = compute_token_amount(result)
        idx += 1

    if (not length):
        write_validation_color(validation_file, "Tokens:" + str(length), "red")
        return ("No more documents could be loaded " +  str(idx)), idx
    
    write_validation_color(validation_file, "Tokens:" + str(length), "")
    write_validation_retriever(validation_file, result)

    return result, idx
    

class Model(ModelBase):    
    def init(self, mode, init):
        def loader(mode):
            loader = get_loader(mode, "**/*.pdf")
            return loader.load_and_split(constants.SPLITTER)
           
        self.retriever = get_retriever("retriever", 
                                       loader, 
                                       mode, 
                                       init, 
                                       constants.DEFAULT_DOC_AMOUNT)

    def getModel(self,validation_file : TextIOWrapper | None):
        return get_custom_retriever(self.retriever, validation_file)
    
    
        
    