import main
import constants


def test_invoke_chain_rest():
    args = constants.DefaultArgs
    args.question = "What's 2+2?"
    result = main.invoke_chain(args)
    print(result)