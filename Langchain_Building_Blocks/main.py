from dotenv import load_dotenv
load_dotenv()
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


from langchain.chat_models import init_chat_model

model = init_chat_model("groq:openai/gpt-oss-120b")

response = model.invoke("What is langchain explain in 2 lines?")

# print("=========Model Response=========")
# print(response.content)



# Prompt Templates
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


#  simple template
simpleTemplate = PromptTemplate(
    input_variables = ["topic"],
    template = "Explain {topic} to a beginner in 2-3 lines."

)

# .format fill the template and returns the final prompt string
formattedPrompt = simpleTemplate.format(topic = "Langchain")
# print("=========Formated Prompt=========")
# print(formattedPrompt)
# print("=========Model Response=========")
# print(model.invoke(formattedPrompt).content)


# Chat Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding tutor."),
    ("human", "Explain {concept} with a simple Java example."),
])

# .format_messages fill the template and returns the list of messages
messages = chat_template.format_messages(concept = "Inheritance in OOPs")

# print("===========Chat messages===========")
# for msg in messages:
#     print(f"{msg.type}: {msg.content}")

# print("\n===========Model Response===========")
# print(model.invoke(messages).content)


# CHAINS

from langchain_core.output_parsers import StrOutputParser
# prompt fills blanks-> model answers -> parses pulls out the clear text
# StrOutputParser just extracts teh plain text from the model's reply
#  don't have to write .content() every time

chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke()

# result = chain.invoke({"concept": "for loops"})
# print("\n========== Chain Output =========")
# print(result)



# MEMORY : Giving the Model Context

# LLMs are stateless, they forget the things between calls
# Memory siply use storing the past messages and feeding them back again
# Modern langchain uses ChatMessageHistory; the old ConversationBufferMemory is depricated and out of the core package.

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import    HumanMessage, AIMessage

# Simple in-memory store, that holds conversations

memory = InMemoryChatMessageHistory()

while(True):
    user_input = input("\nAsk Something: ")

    # exit condition
    if user_input.lower() == "exit":
        break
    memory.add_message(HumanMessage(content=user_input))
    model_response = model.invoke(memory.messages).content.strip()
    print("Model: ")
    print(model_response)
    memory.add_message(AIMessage(content=model_response))