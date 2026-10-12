from dotenv import load_dotenv
import os
load_dotenv()


import warnings
warnings.filterwarnings("ignore", category = UserWarning)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# print(OPENAI_API_KEY)

# Model

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-6-astra")

response = model.invoke("Who is prime minister of India?")

print(response.content)