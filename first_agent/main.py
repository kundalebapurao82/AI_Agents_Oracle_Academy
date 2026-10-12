from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# Initializing the model
model = init_chat_model("groq:openai/gpt-oss-120b")


# define tools

@tool
def multiply(a:float, b: float) -> float:
    """ Multiply two numbers together, it takes two numbers as parameters and returns multiplication of those numbers.
    Used for multiplication operations."""
    return a*b

@tool 
def divide(a: float, b: float) -> float:
    """ Takes two numbers and devide first number by second number. 
    Returns division. If second number (dividor) is zero then it returns error."""
    if b==0:
        return "Error: Can't divide number by zero."
    return a/b

# Creating the langchain Agent
agent = create_agent(
    model=model, 
    tools=[multiply, divide],
    system_prompt= "You are helpful assistant, answer user queries precisely. Use tools whenever required."
     
)

while True:
    
    user_query = input("Ask anything: ")
    if user_query.lower() == "exit":
            print("Bye....")
            break
    result = agent.invoke(
        {
            "messages":[
                {"role": "user", "content": user_query}
            ]
        }
    )
    # print(result['messages'][-1].content)
    print(result)
