from typing import Annotated, Sequence, TypedDict, List, Dict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages.base import messages_to_dict
from langchain_core.load.dump import dumpd, dumps
import asyncio
import json
import uuid

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


llm  = ChatOpenAI(
    base_url=os.getenv("CUSTOM_BASE_URL"),
    api_key=os.getenv("CUSTOM_API_KEY"),
    model_name="/anvme/workspace/unrz103h-helma/base_models/full",
    temperature=0.3,
    streaming=True,
)

MAX_QUERY_COUNT = int(os.getenv("MAX_QUERY_COUNT"))
MAX_MODEL_LENGTH = 3*int(0.85*int(os.getenv("MAX_MODEL_TOKENS")))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY"))

    
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
    result_file: str


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
async def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    if isinstance(state["messages"][-1], HumanMessage):
        # clean previous tools
        messages = [m for m in state["messages"]if not isinstance(m, ToolMessage)]
    else:
        messages = list(state['messages'])
    sys_prompt = fast_answer_system_prompt.format(today=get_today_str())
    messages = [SystemMessage(content=sys_prompt)] + messages
    message = await llm.ainvoke(messages)

    # for Jour fix
    # TODO make this async
    # text, thought = strip_thinking_tokens(message.content)
    # with open(f"JF_deep_research/{state["result_file"]}.md", "w") as f:
    #     f.write(text + "\n")

    state['messages'] = [message]
    return state


# Retriever Agent
async def take_action(state: AgentState) -> AgentState:
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
    state['messages'] = results
    return state

async def write_chat(state: AgentState) -> AgentState:
    """Writes the results of chat into file"""
    messages = list()
    for msg in state["messages"]:
        messages.append(dumpd(msg))

    with open("chat.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": str(uuid.uuid4()), "messages": messages}, ensure_ascii=False)+"\n")

    return state


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_node("writer", write_chat)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: "writer"}
)
graph.add_edge("retriever_agent", "llm")
graph.add_edge("writer", END)
graph.set_entry_point("llm")

app = graph.compile()

QUESTIONS = [
"Wie sieht der Lageplan des Campus Süd in Erlangen aus?",
# "How can students propose changes to university policies through student representation at FAU?",
# "What training or orientation is available for newly elected student representatives at FAU?",
# "Wo finde ich das Vorlesungsverzeichnis der FAU?",
# "Was ist „campo.fau.de“ und wofür wird es genutzt?",
]


semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

async def research(task_id, q):
    async with semaphore:
        await app.ainvoke({"messages": [HumanMessage(q)], "result_file": str(task_id)+"_"+q})

# FILENAME = ""
async def main():
    async with asyncio.TaskGroup() as tg:
        for i, q in enumerate(QUESTIONS):
            try:
                tg.create_task(research(i, q))
                print("Added task", i)
            except Exception as e:
                print("Error - skipped", q)
                print(e)

if __name__ == "__main__":
    asyncio.run(main())
