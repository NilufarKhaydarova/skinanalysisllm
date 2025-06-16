# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY") is not None)
print("HF_TOKEN:", os.getenv("HF_TOKEN") is not None)