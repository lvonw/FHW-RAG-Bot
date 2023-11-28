# AEidI-Aufgabe-1

 Studienordnung mit LLM

 Von Leopold von Wendt, Kim Henrik Otte und Finn Mertens

## Requirements

To properly be able to run this project you need to meet these following
requirements:

- These following files need to be placed into the /data/pdfs/ directory of the
project:
    - Curriculum-B_Inf.pdf
    - Master_Informatik.pdf
    - PVO_2023_V5.pdf
    - ZLO_2021_V2.pdf

- A valid OpenAI API key in a .env file with the keyword 'OPENAI_API_KEY'

### Dependencies

See requirements.txt.

You may directly install all dependencies using pip or anaconda using the
command
```
conda install requirements.txt
```

Or install the following packages:
```
pip install flask langchain python-dotenv pdfplumber pandas tabulate tqdm openai chromadb tiktoken
```

If you added a new dependency you can compile a new list using
```
conda list --export > ./requirements.txt
```

### GUI
If you start the rest.py server, you can interact with the FhWedelChatBot at http://localhost:5000.