import os
from dotenv import load_dotenv
load_dotenv()


CONFIG = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}