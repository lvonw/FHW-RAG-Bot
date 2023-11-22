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
    

def getCustomRetriever(retriever : VectorStoreRetriever):
    k = 1
    lastSearch = None

    @tool
    def search(text : str) -> str:
        """Gets documents with similar content

        Args:
            text: text to search for
        """
        nonlocal k
        nonlocal lastSearch

        k = 1   
        lastSearch = text
        return retriever.get_relevant_documents(text, search_kwargs = {'k': k})[k-1]

    @tool
    def more() -> str:
        """Returns more documents of similar content already searched 
        """

        nonlocal k
        nonlocal lastSearch

        k = k + 1   
        if(k > 4):
            return "more than 4 documents can not be searched"
        if(k > 5):
           raise Exception("more than 5 documents can not be searched")
        
        if lastSearch != None:
            return retriever.get_relevant_documents(lastSearch, search_kwargs = {'k': k})[k-1]
        else:
            return "You need to first search for a text to request more"
    


    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Verwende die Search-Funktion, um ähnliche Textabschnitte zu finden. Wenn der Textabschnitt nicht die gewünschte Antwort liefert, verwende die More-Funktion, um weitere Texte zu finden.",
        ),
        (
            "human","{input}",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
    )

    tools = [more,search]

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

         

def get_Tools(mode=constants.DEFAULT_DATABASE, init=False):

    def loader1(mode):
        return get_loader(mode, "**/*_V*.pdf").load_and_split(constants.SPLITTER)
    tool1 = get_retrieverTool("Prüfungsverordnung", loader1, mode, init)

    def loader2(mode):
        return get_loader(mode, "**/Master*.pdf").load_and_split(constants.SPLITTER)
    tool2 = get_retrieverTool("Modulhandbuch", loader2, mode, init, "Beschreibt Informationen zu den einzelnen Modulen im Master Informatik")

    def loader3(mode):
        return get_loader(mode, "**/Curr*.pdf").load_and_split(constants.SPLITTER)
    tool3 = get_retrieverTool("Studiensverlauf", loader3, mode, init, "Studienverlaufs- und Prüfungsplan Informatik (B.Sc.)")

    return [tool2]
    #return [tool1, tool2, tool3]



# def getModel():
    # prompt = ChatPromptTemplate.from_messages(
    # [
    #     (
    #         "system",
    #         "Du bist ein Assistent zum Beantworten von Fragen für Studierende der FH Wedel",
    #     ),
    #     (
    #         "human","{input}",
    #     ),
    #     MessagesPlaceholder(variable_name="agent_scratchpad")
    # ]
    # )

    # tools = [
    #     more, search
    # ]

    # llm_with_tools = constants.LLM.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    # agent = (
    #     {
    #         "input": lambda x: x["input"],
    #         "agent_scratchpad": lambda x: format_to_openai_function_messages(
    #             x["intermediate_steps"]
    #         ),
    #     }
    #     | prompt
    #     | llm_with_tools
    #     | OpenAIFunctionsAgentOutputParser()
    # )

    # agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # return agent
    
    #====================
    # vectorstore_info = VectorStoreInfo(
    #     name="fh wedel Dokumente",
    #     description="Prüfungsdokumente der FH Wedel",
    #     vectorstore=retriever.vectorstore,
    # )
    # toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=constants.LLM)
    # agent = create_vectorstore_agent(llm=constants.LLM , toolkit=toolkit)
    # 
    #=============#=======
    
    #return agent

class Model(ModelBase):
    def init(self, mode, init):
        def loader2(mode):
            return get_loader(mode, "**/Curr*.pdf").load_and_split(constants.SPLITTER)
            #return get_loader(mode, "**/Master*.pdf").load_and_split(constants.SPLITTER)
        self.retriever = get_retriever("Curr", loader2, mode, init)

    def getModel(self):
        return getCustomRetriever(self.retriever)# customRetrieverQA(self.retriever).getModel()
        
    