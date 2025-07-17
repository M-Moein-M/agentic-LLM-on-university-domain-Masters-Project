from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


import os
import time

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


from open_deep_research.utils import (
    get_today_str,
    strip_thinking_tokens
)

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

load_dotenv()


class Queries(BaseModel):
    queries: List[str] = Field(
        description="List of search queries.",
    )


class AgentState(TypedDict):
    topic: str
    queries: Queries


llm  = ChatOpenAI(
    base_url=os.getenv("CUSTOM_BASE_URL"),
    api_key=os.getenv("CUSTOM_API_KEY"),
    model_name="/anvme/workspace/unrz103h-helma/base_models/full",
    temperature=0.3,
    streaming=True,
)

MAX_QUERY_COUNT = int(os.getenv("MAX_QUERY_COUNT"))
MAX_MODEL_LENGTH = 3*int(0.85*int(os.getenv("MAX_MODEL_TOKENS")))

    
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from open_deep_research.utils import (
    searxng_search,
    es_search
)

from open_deep_research.prompts import fast_answer_system_prompt


@tool
def retriever_tool(queries: List[str]) -> str:
    """
    This tool searches and returns the information from the FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg) website

    Hints to use:
        keep the queries in one single language. Either English or German based on the user message
        Generate max 4 queries. Quality is more importatnt than quanity

    queries:
        A list of SERP optimized queries that retrieve context from FAU website
    """
    # trim extra
    queries = queries[: min(len(queries), MAX_QUERY_COUNT)]

    search_results = searxng_search(queries)
    # search_results = es_search(queries)

    context = ""
    for res in search_results:
        context += f"# {res["query"]}\n"
        for r in res["results"]:
            context += f"## {r["title"]} (source: {r["url"]})\n"
            context += f"{r.get("raw_content", "ERROR 404 - NO CONTENT FOUND FOR THIS PAGE")}"
    
    if len(context) > MAX_MODEL_LENGTH:
        print(f"<<< CUTTING CONTEXT from {len(context)} to {MAX_MODEL_LENGTH} >>>")
        context = context[:MAX_MODEL_LENGTH]

    with open("elastic_result.md", "w") as f:
        print(context, file=f)
        print("written search result .md file")
    return context


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    time: int


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    if isinstance(state["messages"][-1], HumanMessage):
        # clean previous tools
        messages = [m for m in state["messages"]if not isinstance(m, ToolMessage)]
    else:
        messages = list(state['messages'])
    sys_prompt = fast_answer_system_prompt.format(today=get_today_str())
    messages = [SystemMessage(content=sys_prompt)] + messages
    message = llm.invoke(messages)

    # for Jour fix
    with open(f"JF_deep_research/{FILENAME}.md", "w") as f:
        f.write(strip_thinking_tokens(message.content) + "\n")

    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('queries', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

app = graph.compile()

QUESTIONS = [
# for more sample questions look into MONICA_QUESTION.TXT
"Welche Personen gehören zum Kanzleramt der FAU?",
"Wie sieht der Lageplan des Campus Süd in Erlangen aus?",
]


FILENAME = ""
if __name__ == "__main__":

    for i, q in enumerate(QUESTIONS):
        print(f"========= next seed qeustion ========= {i+1}/{len(QUESTIONS)}", q)
        try:
            FILENAME = str(i)+"_"+q
            app.invoke({"messages": [HumanMessage(q)]})
        except Exception as e:
            print("Error - skipped", q)
            print(e)
