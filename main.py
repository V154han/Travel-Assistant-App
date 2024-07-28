# All the imports
from dotenv import load_dotenv
from langchain.tools import Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import login
from transformers import pipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor, LLMSingleActionAgent, create_structured_chat_agent,ZeroShotAgent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
import os

# Initializing the LLM and hugging face
load_dotenv()
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0)
login(os.getenv('HUGGINGFACE_API_KEY'))

# Outlining all the tools
@tool
def search_tool(query : str):
    """Performs online searches to answer the user query."""
    search = DuckDuckGoSearchRun()
    answer = search.run(query)
    return answer

@tool
def summarizaion_tool(query : str):
    """This tool takes a long text and summarizes it"""
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    summary = summarizer(query, max_length=1000, min_length=30, do_sample=False)
    return summary[0]['summary_text']

tools = [
    Tool(name='search_tool', func=search_tool, description="""Performs online searches to answer the user query."""),
    Tool(name = 'summarization_tool', func=summarizaion_tool, description="""This tool takes a long text and summarizes it""")
]

prefix = """Have a conversation with a human, answer the questions as best as you can. You have access to the following tools """
suffix = """
Begin!
{chat_history}
Question: {input}
{agent_scratchpad}
"""


#query = 'What was Apples gross margin in 2022'
#query = 'What is the weather in London today'


#query = 'Give me all the stock allocations for client 3'
prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    input_variables=['input', 'agent_scratchpad', 'tools', 'tool_names', 'history'], 
    prefix=prefix,
    suffix=suffix
   )

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0.2)

llm_chain = LLMChain(llm=llm, prompt=prompt)
memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5)

tool_names = [tool.name for tool in tools]

agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools = tools,
    verbose = True,
)



agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose = True,
    memory = memory
)
