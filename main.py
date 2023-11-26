
from ctypes import ArgumentError
from traitlets import default
from langchain.globals import set_verbose
set_verbose(True)
from base import Logger, colors, get_loader, get_retriever

import os
os.system('color')

# from langchain.globals import set_debug
# set_debug(True)

import constants

import models.tools
import models.vecStore
import models.customTools
import models.smartAgent
import models.assistantApi


from parsing import prepare_arg_parser

def get_answer(answer):
    if isinstance(answer, str):
        return answer
    elif isinstance(answer, dict):
        return answer["output"]
    else:
        raise "ERROR: Answer type " + type(answer)  + "not defined"

def get_user_input():
    return input(constants.INPUT_PROMPT)

validation = [
"Wer unterrichtet Fach Moderne Software Architekturen?", 
"Wie viele ECTS darf man noch offen haben um eine Bachelorthesis anzufangen?",
"Was kannst du übers Auslandssemster sagen?",
"Wie lautet die Modul Nr. von Programmstrukturen 1?",
"Wie lange dauert die Klausur in Programmstrukturen 1?",
"Wie wird ein Workshop in der PVO beschrieben?",
"Empfehle ein Buch für das Modul Algorithmics!",
"Auf welcher Seite der ZLO finde ich Informationen zu BEURLAUBUNG?"]

def main():
    parser = prepare_arg_parser()
    args = parser.parse_args()    
    
    #args.init = True

    # Check if no arguments were provided and print help in that case
    if not args.cli and not args.question and not args.init and not args.validate:
        parser.print_help()
        return
    
    invoke_chain(args)


def invoke_chain(args) -> str:

    log = Logger()

    constants.setLLM(args.gpt)
    modelType : constants.ModelMethod = args.model

    match modelType:
        case constants.ModelMethod.TOOLS:
            model = models.tools.Model()
        case constants.ModelMethod.VECSTORE:
            model = models.vecStore.Model()
        case constants.ModelMethod.CustomTool:
            model = models.customTools.Model()
        case constants.ModelMethod.SMART_AGENT:
            model = models.smartAgent.Model()
        case constants.ModelMethod.OpenAI_ASSISTANT:
            model = models.assistantApi.Model()
        case _: 
            return log.log_red("Model type unknown").output


    #test Docs:
    # def loader2(mode):
    #     return get_loader(mode, "**/Curr*.pdf").load_and_split(constants.SPLITTER)
    #     #return get_loader(mode, "**/Master*.pdf").load_and_split(constants.SPLITTER)
    # retriever = get_retriever("Curr", loader2, args.database, args.init, 10)
    # docs = retriever.invoke("Prüfung in Programmstrukturen 1 Dauer")

    # return


    # Get the correct database and initialize it if demanded
    model.init(args.database, args.init)

    
    if args.validate:
        val_path =  f"validation/validation{args.model}_{args.database}.html"
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        with open(val_path, "w", encoding="utf-8") as file:
            log.set_validation_file(file)
            file.write("<span style=\"white-space: pre\">")#show new line in html!
            
            info = f"Params:\nmodel: {args.model}\ndatabase:{args.database}\nchatGpt:{args.gpt}\n"
            log.log_black(info)
            
            for question in validation:
                log.log_blue( "Frage: " + question)

                chain = model.getModel(file)
                answer = get_answer(chain.invoke(question))

                log.log_green("Antwort: " + answer)
                log.log_black("=============")
            file.write("</span>")#show new line in html!
        return log.output
    
    if not model:
        return
    chain = model.getModel(None)

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
            if hasattr(model, 'retriever'):
                log.log_black(models.vecStore.format_docs(model.retriever.invoke(user_input)))
            else:
                log.log_red("Das Model " + modelType + "besitzt keinen Retriever!")
        else:
            log.log_green("Antwort: " + get_answer(chain.invoke(user_input)))
          
    return log.output


if __name__ == "__main__":
    main()