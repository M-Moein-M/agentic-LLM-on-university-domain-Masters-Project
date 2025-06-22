from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from open_deep_research.prompts import (
    fast_query_writer,
)

from open_deep_research.utils import (
    get_today_str
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


model  = ChatOpenAI(
    base_url="http://10.28.53.143:6000/v1",
    api_key="xFhGltj52Gn",
    model_name="/anvme/workspace/unrz103h-helma/base_models/full",
    temperature=0.3,
    streaming=True,
)

# model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


async def generate_queires(state: AgentState) -> AgentState:
    """ Generate list of targetted SERP queries to gather information to answer topic query

    Args:
        state: current graph state containing the topic
    
    Returns:
        Dict containing list of queries
    """
    print("## STATE:", state)
    topic = state["topic"]
    query_writer = model.with_structured_output(Queries)

    system_instructions_query = fast_query_writer.format(
        topic=topic,
        number_of_queries=4,
        today=get_today_str()
    )
    results = await query_writer.ainvoke(
        [SystemMessage(content=system_instructions_query),
        HumanMessage(content="Generate SERP queries that will help retrieving information for topic.")])
    state["queries"] = results.queries
    print("## STATE:", state)
    
    return state
    

graph = StateGraph(AgentState)

graph.add_node("query_generator", generate_queires)


graph.add_edge(START, "query_generator")
graph.add_edge("query_generator", END)

app = graph.compile()
