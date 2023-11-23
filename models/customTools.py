from base import ModelBase, get_loader, get_retriever, get_retrieverTool
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

from models.tools import getModel

class customRetrieverQA:    
    def __init__(self, retriever : VectorStoreRetriever) -> None:
        self.retriever = retriever
        self.k = 1   

def get_custom_retriever(retriever : VectorStoreRetriever):
    k = 1
    last_search = None

    @tool
    def search(query : str) -> str:
        """Gets documents with similar content

        Args:
            text: text to search for
        """
        nonlocal k
        nonlocal last_search

        k = 1   
        last_sarch = query
        return retriever_call(retriever, query, k)

    @tool
    def more() -> str:
        """Returns more documents of similar content already searched 
        """

        nonlocal k
        nonlocal last_search

        k = k + 1   
        if(k > 4):
            return "more than 4 documents can not be searched"
        if(k > 5):
           raise Exception("more than 5 documents can not be searched")
        
        if last_search != None:
            return retriever_call(retriever, last_search, k)
        else:
            return "You need to first search for a text to request more"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",  constants.AGENT_PROMPT),
        ("human",   "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])

    tools = [more, search]

    llm_with_tools = constants.LLM.bind(functions=[format_tool_to_openai_function(t) for t in tools])

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


def retriever_call(retriever, query, k):
    return retriever.get_relevant_documents(query, search_kwargs = {'k': k})[k-1]



class Model(ModelBase):    
    def init(self, mode, init):    
        self.retriever = get_retriever("Curr", self.__loader, mode, init)

    def getModel(self):
        return get_custom_retriever(self.retriever)
        # return customRetrieverQA(self.retriever).getModel()
    
    def __loader(mode):
        return get_loader(mode, "**/*.pdf").load_and_split(constants.SPLITTER)
        
    