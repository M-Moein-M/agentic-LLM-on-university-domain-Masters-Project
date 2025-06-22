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


llm  = ChatOpenAI(
    base_url="http://10.28.53.143:6000/v1",
    api_key="xFhGltj52Gn",
    model_name="/anvme/workspace/unrz103h-helma/base_models/full",
    temperature=0.3,
    streaming=True,
)


async def generate_queires(state: AgentState) -> AgentState:
    """ Generate list of targetted SERP queries to gather information to answer topic query

    Args:
        state: current graph state containing the topic
    
    Returns:
        Dict containing list of queries
    """
    print("## STATE:", state)
    topic = state["topic"]
    query_writer = llm.with_structured_output(Queries)

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
    
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

load_dotenv()


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg) website
    """

    return "„Wissen bewegen“ ist unser Motto und Leitmotiv an der FAU. Jeden Tag erinnert es uns an die zentrale Rolle und Verantwortung einer Universität in der Gesellschaft. Drei Grundwerte inspirieren, fundieren und leiten dabei unser Denken und Handeln im Alltag: Innovation, Vielfalt und Leidenschaft. Diese finden Sie in unserem Leitbild und in unseren strategischen Handlungsfeldern verankert. Ich lade Sie ein, unsere FAU näher kennenzulernen. Entdecken Sie die Menschen, die unsere Universität ausmachen, die Studienmöglichkeiten und Bildungsangebote, die Forschungsfelder und Forschungskooperationen … Seien Sie neugierig! Wussten Sie beispielsweise schon, dass die FAU die innovationsstärkste Universität in Deutschland ist und europaweit Platz 2 im Ranking der Innovationsführer belegt? Lernen Sie die wichtigsten Innovatoren und Innovationen der FAU kennen. Lassen Sie sich inspirieren. Kommen Sie vorbei und machen Sie mit!"


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions regarding FAU university. Use the retriever tool available to answer questions about the FAU and any topic that canbe related to this university. You can make multiple calls if needed.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
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



# graph = StateGraph(AgentState)

# graph.add_node("query_generator", generate_queires)


# graph.add_edge(START, "query_generator")
# graph.add_edge("query_generator", END)

# app = graph.compile()
