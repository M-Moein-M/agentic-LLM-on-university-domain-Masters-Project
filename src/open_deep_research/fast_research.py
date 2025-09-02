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
    base_url= os.getenv("CUSTOM_BASE_URL"),
    api_key= os.getenv("CUSTOM_API_KEY"),
    model_name= os.getenv("MODEL_NAME"),
    temperature=0.3,
    streaming=True,
)

MAX_QUERY_COUNT = int(os.getenv("MAX_QUERY_COUNT"))
MAX_MODEL_LENGTH = 3*int(0.85*int(os.getenv("MAX_MODEL_TOKENS")))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY"))

url_pool = set()

    
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
async def retriever_tool(queries: List[str]) -> dict:
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

    global url_pool
    search_results, context_urls = await searxng_search(queries)
    url_pool = url_pool.union(context_urls)
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

    return {"context": context, "context_urls": context_urls}


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_id: str
    context_urls: List[str]  # keep track of which urls were used as the context


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
    messages = [SystemMessage(content=fast_answer_system_prompt)] + messages
    message = await llm.ainvoke(messages)

    state['messages'] = [message]
    return state


# Retriever Agent
async def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = await tools_dict[t['name']].ainvoke(t['args'])
            state["context_urls"] = result["context_urls"]

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    state['messages'] = results
    return state


dr_chats = list()

async def flush_chats_to_file() -> None:
    global dr_chats, url_pool
    
    if dr_chats:
        await asyncio.to_thread(_write_chats)
        dr_chats.clear()
    
    if url_pool:
        await asyncio.to_thread(_write_urls)
        url_pool.clear()

def _write_chats():
    with open("./data/chat.jsonl", "a", encoding="utf-8") as f:
        for chat in dr_chats:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

def _write_urls():
    with open("./data/all_urls.txt", "a", encoding="utf-8") as f:
        for url in url_pool:
            print(url, file=f)

async def write_chat(state: AgentState) -> AgentState:
    """Writes the results of chat into file if buffer reaches threshold."""
    global dr_chats, url_pool
    
    messages = [dumpd(msg) for msg in state["messages"]]
    chat = {
        "id": state.get("question_id", "PLACEHOLDER_ID"),
        "messages": messages,
        "context_urls": state.get("context_urls", [])
    }
    dr_chats.append(chat)
    
    if len(dr_chats) >= 5: # TODO change to higher number like 10
        await flush_chats_to_file()

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
"How can students propose changes to university policies through student representation at FAU?",
"What computational methods are taught for solving electromagnetic field problems at FAU?",
"How is innovation fostered at FAU?",
"How does the Chair of Strategic Management at FAU integrate theory and practice in teaching?",
"Wie wird das Thema Digitalisierung an der FAU umgesetzt?",
"How do courses at FAU balance theoretical rigor with practical relevance in computer science?",
"How does FAU prepare students to work in highly automated industrial environments?",
"What role does feature engineering play in physiological signal analysis at FAU?",
"How can international students engage with student representation at FAU?",
"What are the implications of bring-your-own-device (BYOD) policies in academic environments?",
"What are the current trends in hardware design for high performance computing (HPC)?",
"How are innovation and product development taught in the context of industrial engineering at FAU?",
"What is the process for switching majors within the Faculty of Engineering at FAU?",
"How does the Faculty of Engineering at FAU support start-ups and entrepreneurship?",
"How does FAU incorporate global perspectives into its study of Protestant theology?",
]


semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

async def research(task_id, q):
    async with semaphore:
        await app.ainvoke({"messages": [HumanMessage(q)], "question_id": str(task_id)})

# load questions that are not present in chat.jsonl


def load_unanswered_questions(seed_path="./data/seed_questions.jsonl", chat_path="./data/chat.jsonl"):
    """Return list of question dicts from seed_questions.jsonl not present in chat.jsonl, with debug info."""
    answered_ids = set()
    if os.path.exists(chat_path):
        with open(chat_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    answered_ids.add(obj.get("id"))
                except Exception:
                    continue

    questions = []
    total_questions = 0
    skipped = 0
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    total_questions += 1
                    if obj.get("id") not in answered_ids:
                        questions.append(obj)
                    else:
                        skipped += 1
                except Exception:
                    continue
    print(f"[DEBUG] Loaded {total_questions} questions from seed file.")
    print(f"[DEBUG] Found {len(answered_ids)} answered questions in chat file.")
    print(f"[DEBUG] Skipped {skipped} already answered questions.")
    print(f"[DEBUG] {len(questions)} questions will be processed.")
    return questions


async def main():
    questions_to_answer = load_unanswered_questions()
    async with asyncio.TaskGroup() as tg:
        for i, q in enumerate(questions_to_answer):
            try:
                tg.create_task(research(task_id=q["id"], q=q["question"]))
                print("Added task", i)
            except Exception as e:
                print("Error - skipped", q)
                print(e)
    await flush_chats_to_file()  # flush remaining chats

if __name__ == "__main__":
    asyncio.run(main())
