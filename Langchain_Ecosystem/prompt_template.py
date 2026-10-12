from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

simple_template = PromptTemplate(
    input_variables=["topic"],
    template = "Explain the topic {topic} in 2-3 lines to complate beginner."

)

# .format() fills the blank and returns the finished text
formatted = simple_template.format(topic = "Langchain")
print(formatted)



# ChatPromptTemplate: built from roles (system/human), which is how chat
# "System" sets the behavior of the model, "Human" is the user input.

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful coding tutor. Keep aswers short and clear."),
    ("human", "Explain the {concept} with simple Python example."),
])

# .format_messages() fills the blank and returns a list of messages.
messages = chat_template.format_messages(concept = "List Comprehension")

print("Chat messages:")
for msg in messages:
    print(f"[{msg.type}] : {msg.content}")