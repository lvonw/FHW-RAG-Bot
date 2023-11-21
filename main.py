
from traitlets import default
from langchain.globals import set_verbose
set_verbose(True)
# from langchain.globals import set_debug
# set_debug(True)

import constants

import models.tools
import models.vecStore

from parsing import prepare_arg_parser

def get_user_input():
    return input(constants.INPUT_PROMPT)

def main():
    parser = prepare_arg_parser()
    args = parser.parse_args()    
    
    # Check if no arguments were provided and print help in that case
    if not args.cli and not args.question and not args.init:
        parser.print_help()
        return
    
    modelType : constants.ModelMethod = args.model

    match modelType:
        case constants.ModelMethod.TOOLS:
            model = models.tools.Model()
        case constants.ModelMethod.VECSTORE:
            model = models.vecStore.Model()
        case _: 
            print("Model type unknown")
            return

    # Get the correct database and initialize it if demanded
    model.init(args.database, args.init)
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
                print(model.retriever.invoke(user_input))
            else:
                print("Das Model " + modelType + "besitzt keinen Retriever!")
        else:
            print("Antwort: " + chain.invoke(user_input))

if __name__ == "__main__":
    main()