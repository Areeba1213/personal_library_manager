import os
import streamlit as st
import asyncio
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    function_tool,
)

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize external OpenAI-style client (Gemini API)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configure agent run
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define tool
@function_tool
def autism_info_tool(disability_type: str, age: int) -> str:
    if disability_type.lower() == "autism":
        return (
            f"For a {age}-year-old child with Autism:\n"
            "- Use visual schedules\n"
            "- Break tasks into steps\n"
            "- Keep routines predictable\n"
            "- Use AAC tools (like flashcards or pictures)"
        )
    elif disability_type.lower() == "dyslexia":
        return (
            f"For a {age}-year-old child with Dyslexia:\n"
            "- Use multi-sensory learning (sight + sound)\n"
            "- Colored overlays\n"
            "- Extra time for reading tasks\n"
            "- Text-to-speech tools"
        )
    else:
        return (
            f"For a {age}-year-old child with {disability_type}:\n"
            "- Recommend personalized assessment\n"
            "- Consult a special educator\n"
        )

# Create secondary agents
first_agent = Agent(
    name="AI Education",
    instructions="You are a helpful assistant that provides information about artificial intelligence.",
    model=model
)

second_agent = Agent(
    name="AI Assistant Humans",
    instructions="You are a helpful assistant that provides information about humans, including anatomy.",
    model=model
)

third_agent = Agent(
    name="AI Assistant Physics",
    instructions="You are a helpful assistant that provides information about physics.",
    model=model
)

# Create main agent
main_agent = Agent(
    name="Education Assistant",
    instructions="""
    You are a helpful education assistant. You can answer questions related to autism (using the provided tool), artificial intelligence, humans, and physics 
    by handing off to relevant agents. If a user asks about anything outside these topics (like food or chemistry), reply with:
    'Sorry, I cant help with that topic.'
    """,
    tools=[autism_info_tool],
    handoffs=[first_agent, second_agent, third_agent],
    model=model
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Async function to get result
async def get_result(prompt):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    result = await Runner.run(
        main_agent,
        input=st.session_state.chat_history,
        run_config=config
    )
    st.session_state.chat_history.append({"role": "assistant", "content": result.final_output})
    return result.final_output

# Streamlit UI
st.title("🧠 Education Assistant")
st.write("Ask me about autism, AI, humans, or physics!")

prompt = st.chat_input("Ask something...")

if prompt:
    st.chat_message("user").markdown(prompt)
    output = asyncio.run(get_result(prompt))
    st.chat_message("assistant").markdown(output)




















































































