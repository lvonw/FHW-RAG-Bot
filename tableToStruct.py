#konvertiert eine Zeile einer Tabelle in einzelne Json Abschnitte



from typing import List
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from pdfplumber.table import Table
from langchain.llms import OpenAI
import json 
import os
from dotenv import load_dotenv
load_dotenv()
import re

import pdfplumber

def ConvertTable(rows: List[List[str | None]]):

    #1 check title columns:
    first_row = [0 for x in range(len(rows[0]))]

    # 1. find how many rows are header rows?
    headerIndex = 0
    for row in rows:
        found = False
        for ci in range(len(row)):
            if row[ci]:
                first_row[ci] = 1
            if first_row[ci] == 0:
                found = True

        if found:
            headerIndex += 1
        else:
            headerIndex += 1
            break

    # 2. Fix Merged Cells(Text in Merged Cells is represented only in the most left cell)
    for rowi in range(headerIndex):
        row = rows[rowi]
        for ci in range(len(row)-1):
            if rowi > 0:
                rowBefore = rows[rowi-1]
                if not row[ci+1] and (rowBefore[ci + 1] == rowBefore[ci]):
                    row[ci+1] = row[ci]
            elif not row[ci+1]:
                row[ci+1] = row[ci]

    # 3. Combine Header Rows
    for rowi in range(headerIndex):
        row = rows[rowi]
        for ci in range(len(row)):
            if rowi > 0:
                rowBefore = rows[rowi-1]
                if row[ci] and (rowBefore[ci] != rowBefore[ci + 1] if ci < len(row) - 1 else True) and (rowBefore[ci] != rowBefore[ci-1] if ci > 0 else True):
                    rowBefore[ci] += row[ci]
                    row[ci] = "" 

    for row in rows:
        for ci in range(len(row)):
            if row[ci] is not None:
                #remove duplicates:
                words = re.split("[\s,\-\/\n]", row[ci])
                row[ci] = " ".join(sorted(set(words), key=words.index))
                try:
                    row[ci] = float(row[ci])
                except ValueError:
                  pass

    #### Create Json:
    data = []

    startIndex = 1

    for row in rows[headerIndex:]:
        obj = {}
        for ci in range(len(row)):
            if row[ci]:
                ele = obj
                eleObj = obj
                lastIndex = ""
                for i in range(startIndex, headerIndex):
                    if rows[i][ci]:
                        lastIndex = rows[i][ci]
                        if isinstance(ele, dict) and not lastIndex in ele:
                            ele[lastIndex] = {}

                        if isinstance(ele, dict):
                            eleObj = ele
                            ele = ele[lastIndex]
                eleObj[lastIndex] = row[ci]
        data.append(obj)
    return data
