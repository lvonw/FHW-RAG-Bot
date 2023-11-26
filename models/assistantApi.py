from io import TextIOWrapper
from base import ModelBase


class Model(ModelBase):
    def init(self, mode, init):
        pass

    def getModel(self,validate : TextIOWrapper | None):
        pass #Todo return langchain that uses Assistant Api
        
    