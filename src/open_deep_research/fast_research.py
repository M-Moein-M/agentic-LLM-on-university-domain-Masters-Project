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

from open_deep_research.prompts import (
    fast_query_writer,
)

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

load_dotenv()


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
    # search_results = es_search(queries)

    context = ""
    for res in search_results:
        context += f"# {res["query"]}\n"
        for r in res["results"]:
            context += f"## {r["title"]} (source: {r["url"]})\n"
            context += f"{r["raw_content"]}"

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
    messages = [SystemMessage(content=fast_answer_system_prompt)] + messages
    message = llm.invoke(messages)

    # # for Jour fix
    # with open(f"JF_deep_research/{FILENAME}.md", "w") as f:
    #     f.write(strip_thinking_tokens(message.content) + "\n")

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
# "1. Studienangebot und Orientierung",
"Welche Bachelorstudiengänge bietet die FAU an?",
"Gibt es einen Online-Studiengangsfinder oder eine interaktive Suchfunktion?",
"Welche Studiengänge werden in englischer Sprache angeboten?",
"Welche Studiengänge sind zulassungsbeschränkt?",
"Gibt es ein Orientierungsstudium oder eine allgemeine Studienberatung?",
# "📥 2. Bewerbung und Zulassung",
"Wie läuft die Online-Bewerbung für ein Bachelorstudium an der FAU ab?",
"Welche Unterlagen müssen bei einer Bewerbung für ein Masterstudium eingereicht werden?",
"Welche Bewerbungsfristen gelten für das Sommersemester 2026?",
"Wie bewerbe ich mich mit einem ausländischen Schulabschluss?",
"Was ist das „Vorprüfungsdokument“ (VPD) und wo beantrage ich es?",
# "📚 3. Studienorganisation",
"Wo finde ich das Vorlesungsverzeichnis der FAU?",
"Was ist „campo.fau.de“ und wofür wird es genutzt?",
"Wie kann ich ein Urlaubssemester beantragen?",
"Wie funktioniert die Belegung von Seminaren und Übungen?",
"Wo finde ich den akademischen Kalender mit Semesterzeiten und Prüfungsphasen?",
# "🧩 4. Besondere Studienformen und -bedingungen",
"Gibt es Teilzeitstudiengänge an der FAU?",
"Wie unterstützt die FAU Studierende mit Kind?",
"Welche Nachteilsausgleiche gibt es für Studierende mit Behinderung oder chronischer Erkrankung?",
"Gibt es ein „Studium Generale“ oder interdisziplinäre Wahlmöglichkeiten?",
"Welche Möglichkeiten gibt es für ein Doppelstudium oder ein Fachwechsel?",
# "🌍 5. Internationale Studienoptionen",
"Welche Austauschprogramme bietet die FAU an?",
"Welche Partneruniversitäten hat die FAU in Europa?",
"Wie läuft die Anerkennung von im Ausland erbrachten Studienleistungen?",
"Gibt es eine zentrale Beratungsstelle für Outgoing-Studierende?",
"Was müssen internationale Studierende bei der Einschreibung beachten?",
# "🧑‍🤝‍🧑 6. Unterstützung und Leben im Studium",
"Wer berät zu psychischen Belastungen oder Studienzweifeln?",
"Gibt es eine Anlaufstelle für finanzielle Unterstützung, z. B. BAföG oder Notfonds?",
"Wie finde ich Informationen zu Wohnheimen oder privatem Wohnraum?",
"Welche Möglichkeiten für studentisches Engagement gibt es an der FAU?",
"Welche Beratungsstellen oder Programme fördern Diversität und Inklusion?",
]


FILENAME = ""
if __name__ == "__main__":

    for q in QUESTIONS:
        print("========= next seed qeustion =========", q)
        try:
            FILENAME = "_"+q
            app.invoke({"messages": [HumanMessage(q)]})
        except Exception as e:
            print("Error - skipped", q)
            print(e)
