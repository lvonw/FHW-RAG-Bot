from base import ModelBase, get_loader, get_retriever
import constants

from langchain.agents.agent import AgentExecutor
from langchain.schema.vectorstore import  VectorStoreRetriever
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.tools import tool
from langchain.schema import StrOutputParser

from typing import List
from langchain.schema.document import Document

from models.tools import getModel

def get_custom_retriever(retriever : VectorStoreRetriever):
    docs = None 
    idx = 0

    @tool
    def search(query : str) -> str:
        """Gets documents with similar content

        Args:
            text: text to search for
        """
        nonlocal docs
        nonlocal idx
        # Init the documents
        docs = retriever_call(retriever, query, constants.DEFAULT_DOC_AMOUNT)
        res, idx = get_dynamic_doc_amount(docs, idx)
        return res

    @tool
    def more() -> str:
        """Returns more documents of similar content already searched, only
        use if search didnt return enough information to answer the quesiton
        sufficiently 
        """
        nonlocal docs
        nonlocal idx
        
        res, idx = get_dynamic_doc_amount(docs, idx)
        return res
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",  constants.SMART_AGENT_PROMPT),
        ("human",   "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])

    tools           = [more, search]
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

    agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return RunnableLambda(lambda x: {"input" : x}) | agent

def retriever_call(r : VectorStoreRetriever, q, k) -> List[Document]:
    return r.get_relevant_documents(q, search_kwargs = {'k': k})

def compute_token_amount(doc : str):
    return (len(doc) / 4) 

def get_docs_content(doc : Document) -> str:
    return doc.page_content

def get_dynamic_doc_amount(docs : List[Document], startIdx) -> str:
    length = 0 
    idx = startIdx
    result = ""

    while ((idx < len(docs))
            and (length + compute_token_amount(get_docs_content(docs[idx])) 
                < constants.MAX_TOKENS)):   
        
        print(docs[idx].metadata)
        result += get_docs_content(docs[idx])
        idx += 1

    if (result == ""):
        return ("No more documents could be loaded " +  
                str(idx)), idx
    
    return result, idx
    

class Model(ModelBase):    
    def init(self, mode, init):
        def loader(mode):
            return get_loader(mode, "**/*.pdf").load_and_split(constants.SPLITTER)    
        self.retriever = get_retriever("retriever", loader, mode, init, constants.DEFAULT_DOC_AMOUNT)

    def getModel(self):
        return get_custom_retriever(self.retriever)
    
    
        
    