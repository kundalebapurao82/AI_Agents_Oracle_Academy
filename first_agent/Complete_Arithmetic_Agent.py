import math
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KAY")

# Initializing model

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "groq:openai/gpt-oss-120b"
)

# define tools (Helping hands)

from langchain_core.tools import tool

@tool
def add(a:float, b:float) -> float:
    """ Add two numbers together. 
    The agent will use this tool when it detects an addition problem"""
    return a+b

@tool
def subtract(a:float, b:float) -> float:
    """ subtract second number from first number. 
    The agent will use this tool when it detects a subtraction problem"""
    return a-b

@tool
def multiply(a:float, b:float) -> float:
    """ multiply two numbers together. 
    The agent will use this tool when it detects an multiplication problem"""
    return a*b

@tool
def divide(a:float, b:float) -> float:
    """
    Divide first number by second number.
    Agent will use this tool when it detects division problem.
    it returns error when second number (dividor) is zero.
    """

    if b == 0:
        return "Error: number cannot be divided by zero."

    return a/b

@tool
def square_root(num:float) -> str:
    """
    Calculates square root of number.
    includes error hadling for negative numbes.
    """
    if num<0:
        return "Error: Cannot take square root of negative numbers."

    return str(math.sqrt(num))

# Combining all tools into a list
tools = [add, subtract, multiply, divide, square_root]

#  Creating agent
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools= tools,
    system_prompt="You are helpful assistant, response user queries preciesly. Use available tools whenever required."
)



# Run agent
def run_agent():
    """ Run the agent and prints a clean, beginner friendly execution trace"""

    while True:

        user_query = input("Ask anything: ")

        if user_query.lower() == "exit":
            print("Bye.......")
            break

        result = agent.invoke({
            "messages":[{
                "role": "user", "content": user_query
            }]
        })

        print("\nClean Agent Execution Trace..........")

        for msg in result['messages']:
            # Human message: original user message
            if msg.type == 'human':
                print("User: ", msg.content)

            elif msg.type == 'ai'and getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    print(f"I need to use tool: {tool_name}")
                    print(f"Tool input: {tool_args}")

            elif msg.type == "tool":
                print(f"Tool returned : {msg.content}")

            elif msg.type == "ai" and msg.content:
                print(f"AI: {msg.content}")

            print("="*50)


#  Run agent
run_agent()