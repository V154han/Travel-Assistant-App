import streamlit as st
from main import agent_executor
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title='Travel Assistant App', page_icon='✈️', layout='wide')


st.title('Travel Assistant Chatbot')
st.write('''This chatbot is deisgned to help you with all your travel needs.
         It can answer any questions you have regarding your travels as well as summarizing and texts regarding flights!''')


# user_input = st.text_input("You:", "")
# if user_input:
#     response = agent_executor.invoke(user_input)
#     st.text_area("Chatbot:", value=response["output"], height=200, max_chars=None, key=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input(placeholder='Enter your question here')
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role" : "user", "content" : query})

    resp = agent_executor.invoke(query)
    with st.chat_message("assistant"):
        st.markdown(resp['output'])
    st.session_state.messages.append({"role" : "assistant", "content" : resp['output']})
