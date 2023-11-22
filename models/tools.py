from base import ModelBase, get_loader, get_retrieverTool
import constants

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor

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


def getModel(tools):
    agent = initialize_agent(tools, constants.LLM, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True)
    
    #====================
    # vectorstore_info = VectorStoreInfo(
    #     name="fh wedel Dokumente",
    #     description="Prüfungsdokumente der FH Wedel",
    #     vectorstore=retriever.vectorstore,
    # )
    # toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=constants.LLM)
    # agent = create_vectorstore_agent(llm=constants.LLM , toolkit=toolkit)
    # 
    #====================
    
    return agent

class Model(ModelBase):
    def init(self, mode, init):
        self.tools = get_Tools(mode, init)

    def getModel(self):
        return getModel(self.tools)
        
    