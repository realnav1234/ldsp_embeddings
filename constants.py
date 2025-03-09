MODEL = "mpnet"

CUT = 0
if MODEL == "bert":
    CUT = 21
if MODEL == "gpt":
    CUT = 25
if MODEL == "mpnet": 
    CUT = 27