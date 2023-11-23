
from ctypes import ArgumentError
from traitlets import default
from langchain.globals import set_verbose
from base import colors, get_loader, get_retriever
set_verbose(True)
import os
os.system('color')

# from langchain.globals import set_debug
# set_debug(True)

import constants

import models.tools
import models.vecStore
import models.customTools
import models.smartAgent


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
    

    print(invoke_chain(args))


def invoke_chain(args) -> str:
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
        case _: 
            print("Model type unknown")
            return


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
        with open(f"validation{args.model}_{args.database}.html", "w", encoding="utf-8") as file:
            file.write("<span style=\"white-space: pre-line\">")#show new line in html!
            info = f"model: {args.model}\ndatabase:{args.database}"
            file.write(f"<font color=\"blue\">Params:\n{info}</font>\n")
            
            for question in validation:

                print(colors.fg.blue + "Frage: " + question )
                file.write(f"<font color=\"blue\">Frage: {question}</font>\n")

                if hasattr(model, 'retriever'):
                    retr = models.vecStore.format_docs(model.retriever.invoke(question))
                    file.write(f"<font color=\"red\">{retr}</font>\n")
                    print(colors.fg.red + retr)

                chain = model.getModel()
                answer = get_answer(chain.invoke(question))
                print(colors.fg.green + "Antwort: " + answer)
                file.write(f"<font color=\"green\">Antwort: {answer}</font>\n")
                print("=============")
            file.write("</span>")#show new line in html!
        return 
    
    if not model:
        return
    chain = model.getModel()

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
                print(models.vecStore.format_docs(model.retriever.invoke(user_input)))
            if hasattr(model, 'retriver'):
                return model.retriver.invoke(user_input)
            else:
                print("Das Model " + modelType + "besitzt keinen Retriever!")
        else:
            print(colors.bg.green 
                  + "Antwort: " + 
                  get_answer(chain.invoke(user_input)))

if __name__ == "__main__":
    main()