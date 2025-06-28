from typing import Annotated, Sequence, TypedDict, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from open_deep_research.prompts import fast_answer_system_prompt
from open_deep_research.utils import (
    searxng_search,
    get_today_str
)


import os
import time

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from open_deep_research.prompts import (
    fast_query_writer,
    followup_seed_prompt,
    fast_answer_evaluator,
)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Queries(BaseModel):
    queries: List[str] = Field(description="List of search queries.")

class Evaluation(BaseModel):
    evaluation: bool = Field(description="Whether answer fits properly to the queries or not")


llm  = ChatOpenAI(
    base_url=os.getenv("CUSTOM_BASE_URL"),
    api_key=os.getenv("CUSTOM_API_KEY"),
    model_name="/anvme/workspace/unrz103h-helma/base_models/full",
    temperature=0.3,
    streaming=True,
)

MAX_QUERY_COUNT = int(os.getenv("MAX_QUERY_COUNT"))


@tool
def retriever_tool(queries: List[str]) -> str:
    """
    This tool searches and returns the information from the FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg) website

    queries:
        A list of SERP optimized queries that retrieve context from FAU website
    """
    # trim extra
    queries = queries[: min(len(queries), MAX_QUERY_COUNT)]

    search_results = searxng_search(queries)

    context = ""
    for res in search_results:
        context += f"# {res["query"]}"
        for r in res["results"]:
            context += f"## {r["title"]} (source: {r["url"]})\n"
            context += f"{r["raw_content"]}"

    with open("searxng_result.md", "w") as f:
        print(context, file=f)
        print("written search result .md file")
    return context


tools = [retriever_tool]

llm = llm.bind_tools(tools)


def should_continue(state: AgentState) -> Literal["followup_seed", "retriever_agent", END]:
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    if hasattr(result, 'tool_calls') and result.tool_calls:
        return "retriever_agent"
    
    # investigate the quality of response based on the queries
    evaluator = llm.with_structured_output(Evaluation)
    messages = [SystemMessage(content=fast_answer_evaluator)] + list(state['messages'])
    message = evaluator.invoke(messages)
    if message.evaluation:
        return "followup_seed"
    else:
        print("# EVALUATOR: BAD SAMPLE. END")
        return END


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        # clean previous tools
        messages = [m for m in state["messages"]if not isinstance(m, ToolMessage)]
    else:
        messages = list(state['messages'])
    print("## STATE CALL LLM", state)
    messages = [SystemMessage(content=fast_answer_system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


def generate_followup_seeds(state: AgentState) -> AgentState:
    """ generate followup queries to keep learning about the other possible topics """

    query_writer = llm.with_structured_output(Queries)

    
    messages = list(state['messages'])
    messages = [SystemMessage(content=followup_seed_prompt)] + messages
    message = query_writer.invoke(messages)

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
graph.set_entry_point("llm")

graph.add_node("llm", call_llm)
graph.add_node("followup_seed", generate_followup_seeds)
graph.add_node("retriever_agent", take_action)

graph.add_edge("retriever_agent", "llm")
graph.add_edge("followup_seed", END)
graph.add_conditional_edges("llm",should_continue)

app = graph.compile()
