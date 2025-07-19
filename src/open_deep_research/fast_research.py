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

    global url_pool
    search_results, all_urls = searxng_search(queries)
    for u in all_urls:
        url_pool.add(u)
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

    with open("chat.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": str(uuid.uuid4()), "messages": messages}, ensure_ascii=False)+"\n")
    global url_pool
    with open("all_urls.txt", "a", encoding="utf-8") as f:
        for url in url_pool:
            print(url, file=f)
    url_pool = set()

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
"Why is long‑term digital archiving of academic publications important for research visibility?",
"How do courses at FAU balance theoretical rigor with practical relevance in computer science?",
"How does FAU prepare students to work in highly automated industrial environments?",
"What role does feature engineering play in physiological signal analysis at FAU?",
"What are the research opportunities for master's students at the Chair of Information Systems at FAU?",
"What tools or advisory services are commonly offered by university libraries to promote open access and repository use?",
"What are the main challenges of providing IT services to a large university community?",
"How does the Chair approach training in modern topics like physics‑informed neural networks or operator learning?",
"Wie unterstützen die Lehrvideos des Labors das Verständnis von Mustererkennungsmethoden?",
"How can students at FAU gain hands-on experience with program verification?",
"How can researchers deposit scientific works in an open access repository without paying fees?",
"Wie fördert die Lab‑Typ‑Galerie (PhD Gallery) den Einblick in Promotionsprojekte?",
"How does FAU support international students in the industrial engineering and management program?",
"Wie ist die FAU historisch gewachsen?",
"Wie ist die FAU organisatorisch aufgebaut?",
"What types of services does the university library at FAU offer to new students?",
"How does the School of Business, Economics and Society at FAU support international students during their studies?",
"How are time-domain and frequency-domain methods taught and applied at FAU?",
"Was versteht die FAU unter dem Konzept des lebenslangen Lernens?",
"How is the relationship between faith and reason explored in Protestant theology at FAU?",
"What challenges in data interoperability are studied in smart production at FAU?",
"How is artificial intelligence research conducted within the Faculty of Engineering at FAU?",
"What is the structure of the doctoral programs offered at the Faculty of Medicine at FAU?",
"Welche Ansätze verfolgt das Lehrstuhl für Mustererkennung zur automatischen Klassifikation von Bilddaten?",
"What simulation tools are used in digital factory research at FAU?",
"How do computing centers manage access control and authentication for online university services?",
"What advanced topics are covered in graduate-level electrical engineering courses at FAU?",
"How does FAU prepare engineering students for global job markets?"
"What is the function of job queuing systems like SLURM in high performance computing (HPC)?",
"What soft skills are emphasized in the industrial engineering and management program at FAU?",
"Wie arbeitet das Pattern Recognition Lab mit internationalen Forschungseinrichtungen zusammen?",
"How are students evaluated in the Medical Engineering program at FAU?",
"What are the key challenges in integrating artificial intelligence into production systems studied at FAU?",
"What challenges do student representatives face at FAU?",
"What training opportunities are available for students interested in robotics at FAU?",
"What resources are available for learning German academic terminology through the library of FAU?",
"What are the benefits of studying term rewriting systems at FAU?",
"What strategies are used by student representatives to increase student participation in university governance at FAU?",
"What topics are covered in foundational courses on abstract machines at FAU?",
"What interdisciplinary opportunities are available for engineering students at FAU?",
"What are some interdisciplinary research projects currently active in industrial engineering at FAU?",
"What interdisciplinary programs are available at the Faculty of Medicine at FAU?",
"What case studies illustrate the use of control and machine learning for physical or engineering systems?",
"How are virtual desktop infrastructures used in academic institutions?",
"What frameworks are used at FAU to evaluate the efficiency of energy markets?",
"What are typical topics covered in seminars on strategic management at FAU?",
"How can prospective PhD students find supervisors in the medical faculty at FAU?",
"What is the historical development of the Faculty of Law at FAU?",
"How can alumni of the Faculty of Law at FAU stay connected and involved?",
"What strategies are used to maintain and upgrade campus-wide Wi-Fi infrastructure?",
"What are the pedagogical goals and main teaching methods used in Ferienintensivkurs (intensive vacation German courses)?",
"How are clinical applications integrated into the FAU Medical Engineering program?",
"Wie werden Sprachverarbeitung und Sprachverständnis im Kontext der Mustererkennung erforscht?",
"What is the significance of Hoare logic in the study of software correctness at FAU?",
"How are students trained to assess the economic viability of engineering solutions at FAU?",
"How are carbon pricing mechanisms examined in FAU's energy market courses?",
"What collaborations exist between the Institute for Empirical Education and Social Science and other faculties at FAU?",
"How do law students at FAU contribute to public discourse or community legal education?",
"What insights does FAU provide into the economic feasibility of decentralized energy systems?",
"Welche englischsprachigen Studiengänge bietet die FAU an?",
"What are the long-term research goals of the industrial engineering and management department at FAU?",
"How are methods for measuring expectation uncertainty used in research by the Institute of Economic Research?",
"What are the major bottlenecks in high performance computing (HPC) workflows?",
"What is the importance of economics in the industrial engineering and management curriculum at FAU?",
"How is the impact of secularization on religious belief discussed at FAU?",
"How does the School of Business, Economics and Society at FAU collaborate with companies and industry partners?"
"How are sustainability and environmental concerns addressed in industrial engineering education at FAU?",
"What is the role of legal philosophy in the curriculum of FAU’s law faculty?",
"How does the School of Business, Economics and Society at FAU foster entrepreneurship and innovation among students?",
"What role does digitalization play in the industrial engineering and management curriculum at FAU?",
"What are the ethical considerations in using AI for patient monitoring at FAU?",
"How does FAU teach the design and implementation of mechatronic systems?",
"How does the Machine Learning and Data Analytics Lab at FAU collaborate with hospitals or clinical institutions?",
"What are the most active research groups within the Faculty of Engineering at FAU?",
"What are the key components of an effective university-wide learning management system?",
"What mechanisms support innovation transfer at FAU?",
"What are some challenges in modeling electromagnetic systems discussed at FAU?",
"What makes the Department of Computer Science 1 at FAU a strong place for theory-focused students?",
"How does the Medical Engineering program at FAU relate to biomedical engineering in terms of content and focus?",
"What collaborations exist between student representatives at FAU and other German universities?",
"How are lifecycle considerations incorporated into mechatronic system design at FAU?",
"What degree programs are offered by the Faculty of Engineering at FAU?",
"How does FAU accommodate students with disabilities in law programs?",
"What foundational algorithms are studied in the Artificial Intelligence programs at FAU?",
"Welche Unterstützung bietet die FAU für internationale Studierende?",
"What digital tools and platforms are used for teaching at the School of Business, Economics and Society at FAU?",
"What mechanisms exist to ensure transparency in student representation at FAU?",
"What are the major research areas in energy markets at FAU?",
"Wie unterstützt das LME industrielle Partner bei Anwendungen im Bereich Health Engineering?",
"How do alumni of FAU’s industrial engineering and management program contribute to the university?",
"What are the main responsibilities of a network operations team in a university setting?",
"How are electromagnetic compatibility and signal integrity studied at FAU?",
"What role do alumni play in the development of FAU?",
"How does FAU explore the theological implications of digital and media culture?",
"How does FAU support interdisciplinary research?",
"What kind of support do students receive when learning abstract computer science topics at FAU?",
"What role does sensor integration play in automation solutions at FAU?",
"How is power electronics integrated into the curriculum at FAU?",
"What is the role of containerization in high performance computing (HPC)?",
"How does the Alexander von Humboldt Professorship support the development of research at this Chair?",
"Wie fördert die FAU die Chancengleichheit in der Wissenschaft?",
"How is the concept of Industry 4.0 integrated into FAU's industrial engineering curriculum?",
"How does the Chair of Information Systems at FAU support lifelong learning in the digital economy?"
"What role do augmented reality and virtual reality play in factory planning at FAU?",
"How is student feedback used to improve chemistry education at FAU?",
"What are the key challenges in integrating intermittent renewable sources in energy markets studied at FAU?",
"How does FAU encourage responsible AI development in its programs?",
"What are the typical learning outcomes for students completing the industrial engineering and management degree at FAU?",
"What are the main research themes at the Institute for Empirical Education and Social Science at FAU?",
"What are the advantages of using federated identity systems in academic IT environments?",
"What strategies are used for reducing overfitting in biomedical machine learning models at FAU?",
"How does FAU investigate the transition to low-carbon energy systems?",
"How can international students contribute to and benefit from the work of the Chair of Information Systems at FAU?",
"Why mirror privacy‑focused tools like Tails or OSMC and what are their use cases?",
"How are wearable systems validated for clinical use at FAU?",
"How does the FAU industrial engineering program prepare students for leadership roles in industry?",
"What is the process for renewing borrowed items at the university library of FAU?",
"How does the Department of Computer Science 1 at FAU integrate formal proofs into its coursework?",
"What consulting services does the Institute for Empirical Education and Social Science provide in empirical research methods?",
"Wie integriert das Pattern Recognition Lab Reinforcement Learning in seine Forschungsprojekte?",
"How does the Institute of Economic Research analyze the transformation from fossil fuels to hydrogen in energy markets?",
"What is the role of structural operational semantics in programming language education at FAU?",
"How does the Institute for Empirical Education and Social Science support evidence‑based decision making in educational settings?",
"How is teaching organized across different faculties at FAU?",
"What is the role of the Comprehensive Cancer Center Erlangen-EMN in medical education at FAU?",
"What funding is available to support student-led initiatives through student representation at FAU?",
"What are the main research areas of the Machine Learning and Data Analytics Lab at FAU?",
"What are the key economic principles that underpin energy market design taught at FAU?",
"What ethical issues are addressed in the legal training at FAU?",
"What role does systems engineering play in production systems design at FAU?",
"What support is available for students interested in joining student politics at FAU?",
"What is the impact of centralized IT governance on service delivery in universities?",
"What is the structure of student representation at FAU?",
"In what ways does the Institute for Empirical Education and Social Science support the Master’s programme in Educational Research on Learning and Instruction at FAU?",
"What challenges arise in managing personal data and privacy in university IT systems?",
"What challenges do students face when learning proof theory at FAU and how are they supported?",
"How does mirroring benefit local clients in terms of download speed and reliability?",
"Welche Strategien verfolgt die FAU zur Förderung der Nachhaltigkeit?",
"How does FAU foster innovation and entrepreneurship in chemical sciences?",
"In what way can mirror statistics inform researchers about digital distribution trends?",
"How does a mirror synchronise package data from upstream repositories?",
"How do student representatives collaborate with faculty and staff at FAU?",
"How are market failures in energy systems addressed in FAU research?",
"What kinds of data backup and storage services are essential for students and researchers?",
"What role does user-centered design play in the innovation process taught at FAU’s Chair of Information Systems?",
"What are the core principles of lean automation covered in FAU's curriculum?",
"How can students at FAU get involved in research projects related to digital innovation and entrepreneurship?",
"How does FAU integrate artificial intelligence in chemical and pharmaceutical research?",
"How does the Chair of Information Systems at FAU assess the scalability of digital business models?",
"How does FAU investigate the social and economic impacts of energy transition policies?",
"How are students involved in shaping academic programs at FAU’s Faculty of Engineering?",
"What is the historical development of FAU since its founding?",
"What role does Protestant theology play in shaping educational policy discussions at FAU?",
"What are the major areas of specialization available in the law programs at FAU?",
"How are advanced algorithms introduced and applied in FAU computer science programs?",
"How does an open access repository assign persistent identifiers or DOIs to archived works?",
"What interdisciplinary collaborations exist between computer science and medicine at FAU?",
"How is the economics of nuclear power discussed in FAU's energy courses?",
"How is the relationship between electromagnetic theory and circuit theory emphasized at FAU?",
"What are the differences between public law and private law tracks at FAU?",
"What are the key focus areas of the Chair of Information Systems – Innovation and Value Creation at FAU?",
"How does FAU handle data privacy and ethical concerns in medical AI research?",
"What research methods are commonly used at the Chair of Information Systems – Innovation and Value Creation at FAU?",
"Wie werden Audiodaten zur Diagnostik oder Gesundheitsanalyse im LME genutzt?",
"What are the differences between print and electronic resources available at the FAU library?",
"What role does digitalization play in production systems research at FAU?",
"How does the university library of FAU support research data management?",
"Wie implementiert das LME bildbasierte Deep‑Learning‑Modelle in Infrastruktur‑Kontexten?",
"What are the benefits of using subject librarians at the university library of FAU?",
"How is machine learning applied to factory automation research at FAU?",
"How might students or staff contribute to or propose new mirror services at FAU?",
"What is the impact of FAU’s research in electromagnetic simulation tools?",
"What is the importance of formal methods in computer science education at FAU?",
"What kind of hardware resources (e.g., GPUs, robotics kits) are available to AI students at FAU?",
"What support services are available to first-year law students at FAU?",
"How does FAU evaluate student performance in project-based industrial engineering courses?",
"How does the Institute of Economic Research support graduates pursuing careers in economic research or policy?",
"How does FAU prepare Medical Engineering students for doctoral research or academic careers?",
"Welche Forschungsinstitute sind der FAU angegliedert?",
"What software platforms are used for factory automation research at FAU?",
"How does the Institute for Empirical Education and Social Science integrate interdisciplinary perspectives in education research?",
"How is unsupervised learning applied in health-related research at FAU?",
"How does the FAU Department of Chemistry and Pharmacy collaborate with industry partners?",
"What kinds of digital identity services are typically offered by a university computing center?",
"How does student representation at FAU support student clubs and organizations?",
"What research institutes are affiliated with FAU?",
"What are the unique features of the Electrical Engineering and Information Technology program at FAU?",
"What strategies does FAU use to recruit top academic talent?",
"What kinds of theses are typically written by students at the Chair of Information Systems at FAU?",
"What are typical use cases of high performance computing (HPC) in industry?",
"How do industrial engineering students at FAU stay updated with technological advances in manufacturing?",
"How is the Faculty of Engineering at FAU structured in terms of departments?",
"What ethical issues are discussed in industrial engineering and management courses at FAU?",
"What are the key differences between undergraduate and graduate Artificial Intelligence programs at FAU?",
"How is the concept of market design for energy systems introduced to students at FAU?",
"How does the FAU Medical Engineering program handle interdisciplinary teaching between engineering and medical sciences?",
"How does FAU study the role of religion in historical and contemporary conflicts?",
"How does the School of Business, Economics and Society at FAU integrate practical business cases into academic teaching?",
"What support services (e.g. language learning advising, writing centre, translation service) are available to German learners at FAU?",
"What graduate-level specializations are offered in chemistry at FAU?",
"What opportunities exist for interdisciplinary master's programs at FAU?",
"How are fairness and bias in AI addressed in the curriculum at FAU?",
"How do university computing centers support research projects with specialized IT needs?",
"What are the admission requirements for international students applying to the School of Business, Economics and Society at FAU?",
"What training or orientation is available for newly elected student representatives at FAU?",
"How can students apply for funding or scholarships at the Faculty of Medicine at FAU?",
"What career services are provided for students at the School of Business, Economics and Society at FAU?",
"How can university computing centers encourage digital literacy among students and staff?",
"What statistical and econometric tools are used in energy market research at FAU?",
"What is the approach to teaching chemical safety and regulation at FAU?",
"How does the FAU Medical Engineering program integrate knowledge from mechanical and electrical engineering?",
"How are conflicts between students and university staff handled through student representation at FAU?",
"How does FAU evaluate the economic impact of automation technologies?",
"How does FAU’s Faculty of Engineering promote sustainability in research and education?",
"What are the common challenges faced by new users of high performance computing (HPC) systems?",
"What are the ethical implications of using high performance computing (HPC) in research?",
"What laboratory facilities are available to undergraduate chemistry students at FAU?",
"What kinds of data are typically mirrored (e.g. Linux distributions, applications, datasets)?",
"What is the significance of Martin Luther’s theology in the curriculum at FAU?",
"What is the application process for the Master’s program in Medical Process Management at FAU?",
"How does FAU approach the development of autonomous manufacturing systems?",
"How is spectroscopy used in student research at FAU?",
"How does the FAU Medical Engineering program address current trends in digital health?",
"How does FAU address the societal impact of Artificial Intelligence in its academic programs?",
"What are the interdisciplinary applications of control and numerics in engineering, physics, biology or social sciences?",
"How does FAU incorporate real-world business cases into its industrial engineering education?",
"What role does the student parliament play at FAU?",
"How does FAU’s Faculty of Engineering support female students in STEM fields?",
"How does FAU address challenges in real-time data processing from wearable devices?",
"What options exist for specialization within the Artificial Intelligence study programs at FAU?",
"How do students at FAU participate in industry-related projects during their studies?",
"How does the Chair of Information Systems at FAU prepare students for careers in digital consulting?",
"What are the main research areas in Protestant theology at FAU?",
"How is medical ethics taught at the Faculty of Medicine at FAU?",
"How is software licensing handled for educational and research purposes at a university level?",
"How does the Faculty of Law at FAU support career development for law graduates?",
"What courses at FAU focus on antenna design and propagation?",
"How does the Institute for Empirical Education and Social Science approach research synthesis in education studies at FAU?",
"How can students at FAU get involved in research on formal verification techniques?",
"What are the opportunities for students to engage in social or community projects at the School of Business, Economics and Society at FAU?",
"How is the topic of human-machine interaction covered in industrial engineering at FAU?",
"What types of theses are typically supervised at the Chair of Strategic Management at FAU?",
"How does FAU integrate interdisciplinary approaches in its Artificial Intelligence curriculum?",
"What tools can be used to visualise mirror traffic and access patterns?",
"Wie werden multimodale Daten zur Verbesserung der Diagnostik genutzt?",
"What foundational programming paradigms are emphasized in the computer science curriculum at FAU?",
"How is mathematics used in the Artificial Intelligence courses at FAU?",
"How are control systems designed and optimized in FAU’s automation studies?",
"Welche Rolle spielt die FAU in der regionalen Entwicklung?"
"What is the mission of the Institute of Economic Research at FAU?",
"What storage technologies (RAID, NVMe, LVM) are suitable for large mirror volumes?",
"How does FAU measure research impact?",
"What pedagogical or mentoring support does the Chair offer to PhD students and postdoctoral researchers?",
"What frameworks are used to analyze digital platform strategies at FAU’s Chair of Information Systems?",
"How are chemistry teaching methods adapted to current scientific trends at FAU?",
"What research topics are covered by the Chair of Empirical Microeconomics at the Institute of Economic Research?",
"What theological approaches to ethics are studied at FAU?",
"How do national and regional collaborations influence high performance computing (HPC) research?",
"Wie ist die Lehramtsausbildung an der FAU organisiert?",
"What recent trends in strategic management research are being explored at FAU?",
"How is electromagnetic field theory applied in research at FAU?",
"What tools and languages are used in formal method courses at FAU?",
"What mentoring programs are available for new engineering students at FAU?",
"What are the benefits and limitations of using smartphone sensors for health monitoring at FAU?",
"What are common career paths for students specializing in strategic management at FAU?",
"What role does the Institute of Economic Research play in energy economics research at FAU?",
"What ethical frameworks are introduced in the Artificial Intelligence programs at FAU?",
"How does FAU support innovation in the field of production engineering?",
"How does FAU integrate artificial intelligence in industrial engineering education?",
"What are the core subjects covered in the first year of industrial engineering and management at FAU?",
"How do university IT services ensure accessibility for students with disabilities?",
"How are students involved in public health policy projects at FAU?",
"What types of machine learning methods are most commonly used in biomedical signal processing at FAU?",
"What are common prerequisites for enrolling in Artificial Intelligence programs at FAU?",
"What are the expectations for master's theses in strategic management at FAU?",
"What are the most common IT issues faced by students and how are they resolved?",
"What is the balance between theoretical and applied learning in the Medical Engineering curriculum at FAU?",
"What are the rights and responsibilities of a student representative at FAU?",
"What languages are used for instruction in the Artificial Intelligence programs at FAU?",
"How can doctoral and habilitation theses be submitted and managed within an institutional repository?",
"What theological perspectives on social justice are taught at FAU?",
"How do student representatives influence university policy at FAU?",
"How do chemistry students at FAU gain experience with research publications?",
"What are the attendance requirements and consequences for not meeting attendance policies in German courses at FAU?",
"What are the major challenges in applying machine learning to medical diagnostics at FAU?",
"What steps and documentation are required from international students to enroll in preparatory DSH courses?",
"How are societal impacts of digital innovation evaluated at the Chair of Information Systems at FAU?",
"How does FAU ensure quality assurance in teaching?",
"What interdisciplinary collaborations exist between economics and engineering in energy studies at FAU?",
"What kind of thesis topics are common in industrial engineering and management at FAU?",
"Wie nutzt das Labor Deep Learning für CT‑(Computertomographie) oder Röntgenbildverarbeitung?",
"What strategies are used at FAU to teach advanced concepts in dataflow analysis?",
"What are the primary components of a high performance computing (HPC) system?",
"How can Artificial Intelligence students at FAU contribute to open-source projects?",
"What practical skills are emphasized in the legal education at FAU’s Faculty of Law?",
"How does FAU ensure research integrity and ethics?",
"What competencies do students gain from participating in seminars at the Chair of Information Systems at FAU?",
"How is deep learning used to analyze gait patterns at FAU?",
"How is Artificial Intelligence applied to engineering problems in courses at FAU?",
"What role do collaborative robots play in FAU's production engineering research?",
"Welche Möglichkeiten zur interdisziplinären Zusammenarbeit bestehen für Forschende an der FAU?",
"How does the Faculty of Law at FAU structure its Bachelor of Laws (LL.B.) program?",
"What software tools are commonly used in electrical engineering research at FAU?",
"Welche Möglichkeiten zur Weiterbildung gibt es an der FAU für Berufstätige?",
"What support structures exist for international students studying Artificial Intelligence at FAU?",
"What is the role of accelerators like tensor processing units (TPUs) in high performance computing (HPC)?",
"How does the Chair ensure the integration of scientific computing with real‑world problem domains?",
"How do FAU instructors approach teaching computational complexity theory?",
"How can international students access resources at the university library of FAU?",
"How is the master's program in electrical engineering at FAU structured?",
"How are neural networks taught and applied in the Artificial Intelligence courses at FAU?",
"How does the Institute of Economic Research analyze health economics topics at FAU?",
"What are the most important skills Medical Engineering students develop during their studies at FAU?",
"How does FAU address the role of religion in public life through its religious studies programs?",
"How is nanotechnology research conducted within FAU’s Faculty of Engineering?",
"What role does FAU play in advancing digitalization?",
"What is a software mirror server and why do universities operate them?",
"How does the existence of redundant mirror nodes improve fault tolerance?",
"What topics are covered in courses related to digital business models at the Chair of Information Systems at FAU?",
"What role does digitalization play in the research strategy of FAU’s Faculty of Engineering?",
"What is the significance of the Cluster of Excellence at FAU?",
"What software tools are commonly used in strategic management courses at FAU?",
"How does FAU promote interfaith dialogue within its religious studies programs?",
"What role does environmental chemistry play in research and teaching at FAU?",
"What are the academic and professional qualifications of the team members working at the DCN‑AvH Chair?",
"What are the admission requirements for studying law at FAU?",
"How does the Faculty of Medicine at FAU approach personalized medicine in education and research?",
"How are hybrid production systems designed and analyzed at FAU?",
"How does FAU examine eschatology within Protestant theology?",
"How does FAU analyze the interaction between energy markets and financial markets?",
"What opportunities exist for students at FAU to publish their machine learning research?",
"What is the role of logistics in the industrial engineering and management education at FAU?",
"What role do electrocardiogram (ECG) signals play in machine learning research at FAU?",
"How is exam performance evaluated in the law programs at FAU?",
"What facilities are available for simulation-based medical training at FAU?",
"How are German phonetics and pronunciation addressed within the curriculum and available resources for learners?",
"What collaborations does FAU have with international institutions in electrical engineering?"
"How can students get involved in university-level committees at FAU?",
"How does the Institute of Economic Research engage in externally funded projects in applied economics?",
"What key concepts are emphasized when introducing students to rewriting logic at FAU?",
"How are students introduced to working with time-series biomedical data at FAU?",
"What are incoming and outgoing traffic in the context of a mirror service?",
"How can students combine medical studies with research in neuroscience at FAU?",
"What international exchange programs are available to engineering students at FAU?",
"How does a university computing center ensure secure and reliable campus network infrastructure?",
"How are electricity grid constraints modeled in FAU’s energy market simulations?",
"How does the Chair of Information Systems at FAU define and measure digital transformation success?",
"What is the importance of scalability in high performance computing (HPC) applications?",
"How is historic archival content preserved in long‑term mirror infrastructures?",
"How is virtual commissioning used in production system design at FAU?",
"What are best practices for time-efficient academic research using the FAU library?",
"How are theological ethics applied to bioethical questions in FAU’s research?",
"What are the research strengths of FAU in engineering?",
"What types of law-related student organizations are active at FAU?",
"How does FAU address the relevance of religion in postmodern societies?",
"How is fault tolerance implemented in high performance computing (HPC) systems?",
"How does the Chair in Dynamics, Control, Machine Learning and Numerics contribute to open‑source or publicly available research software?",
"What is the significance of interdisciplinary teaching in shaping the future of industrial engineering at FAU?"
"What kinds of experimental techniques are emphasized in FAU’s chemistry laboratories?",
"How do student representatives influence decisions at the School of Business, Economics and Society at FAU?",
"How does the Faculty of Engineering at FAU collaborate with other faculties?",
"What empirical research methods are commonly taught and used at the Institute for Empirical Education and Social Science?",
"What tools are available at the FAU library to help students manage citations and references?",
"How does open access repository interoperability with platforms like OpenAIRE or national libraries work?",
"What is the approach to ethics and corporate responsibility in strategic management courses at FAU?",
"What are typical thesis topics in the Machine Learning and Data Analytics Lab at FAU?",
"What is the role of the Institute of Radiology in student education at FAU?",
"What are the major research areas in electrical engineering at FAU?",
"How does FAU prepare Medical Engineering students for interdisciplinary teamwork?",
"What programming languages and libraries should students know before joining biomedical AI projects at FAU?",
"What career services are available for medical graduates at FAU?",
"What are the benefits of studying at FAU for international students?",
"What are the main differences between industrial engineering and traditional mechanical or electrical engineering?",
"How does the Faculty of Medicine at FAU engage with the regional healthcare system?",
"What facilities and laboratories are available for engineering students at FAU?",
"Wie werden neue Studierende an der FAU unterstützt?",
"What international exchange programs are available for students at the School of Business, Economics and Society at FAU?",
"What is the role of student representatives in promoting diversity and inclusion at FAU?",
"What kinds of interdisciplinary collaboration exist between the Faculty of Medicine and other faculties at FAU?",
"How are student voices included in academic policy decisions at FAU?",
"What training sessions or workshops does the university library of FAU offer to students?",
"How is biblical interpretation taught and researched within Protestant theology at FAU?",
"What support does FAU provide for chemistry students planning academic careers?",
"How do computing centers collaborate with academic departments to support teaching and learning?",
"What frameworks are commonly introduced in FAU's strategic management courses?",
"What is the language of instruction for the law courses at FAU?",
"How is the integration of humanities and technology approached at FAU?",
"How is the role of capacity markets analyzed in FAU’s academic studies?",
"How is industry collaboration integrated into the engineering programs at FAU?",
"What language requirements exist for studying engineering programs at FAU?",
"How does preparation for the TestDaF (Test Deutsch als Fremdsprache) differ from preparation for the DSH examination?",
"In what ways are learning and instruction theories tested empirically at FAU’s Institute for Empirical Education and Social Science?",
"How are energy storage solutions analyzed in relation to market dynamics at FAU?",
"How does the School of Business, Economics and Society at FAU help students connect with potential employers?",
"How do mirror networks support large scientific events by distributing content?",
"What academic disciplines are covered at FAU?",
"Welche Rolle spielen bekannte Operatoren (known operators) in den Lernverfahren des Labors?",
"How can students at FAU get involved in research projects related to strategic management?",
"What is the role of machine learning in electrical engineering research at FAU?",
"Wie adressiert das Labor inverse Probleme in der medizinischen Bildgebung?",
"How are email services managed and secured in a university computing environment?",
"How does one access and utilize a national high performance computing (HPC) center?",
"How are emerging topics like quantum electromagnetics addressed in FAU’s programs?",
"How does the Faculty of Engineering engage with the Erlangen-Nuremberg metropolitan region?",
"What teaching responsibilities does the Institute of Economic Research have for bachelor’s and master’s programmes?",
"How do compiler optimizations influence the performance of high performance computing (HPC) applications?",
"How does FAU support open science initiatives?",
"How does FAU support international students in the Medical Engineering program?",
"How can students build a portfolio of AI-related projects while studying at FAU?",
"What operational challenges arise when supporting mirrors for multiple Linux distributions?"
"How are modern communication systems studied within the electrical engineering programs at FAU?",
"How are ethical concerns in pharmaceutical development addressed at FAU?",
"What collaborations exist between FAU Medical Engineering and hospitals or clinics?",
"How do authors verify publisher policies to ensure self‑archiving is allowed in a repository?",
"How does FAU explore the role of religion in contemporary ethical debates?",
"What support is available for students who want to launch AI-related startups at FAU?",
"What is the role of church history in the academic formation of theology students at FAU?",
"How does FAU support early career researchers in the medical field?",
"How is strategic bidding behavior in electricity markets studied at FAU?",
"What copyright considerations should students be aware of when using library resources at FAU?",
"How is the historical development of the Christian church analyzed at FAU?",
"What are the safety considerations in autonomous production systems taught at FAU?",
"What role does computer science play within the Faculty of Engineering at FAU?",
"What support services are available for international medical students at FAU?",
"What committees at FAU include student representatives?",
"What kinds of group projects are part of the strategic management curriculum at FAU?",
"How does high performance computing (HPC) support research in climate modeling?",
"Welche Strategien verfolgt das Labor bei der Datenvorverarbeitung für medizinische Big Data?",
"What is the role of religious education in the curriculum of Protestant theology at FAU?",
"What support does the university library at FAU provide for writing theses or dissertations?",
"What is the relationship between student government and the university administration at FAU?",
"What learning formats are used in Artificial Intelligence courses at FAU (e.g., lectures, labs, seminars)?",
"What double-degree programs are available at the School of Business, Economics and Society at FAU?",
"What is the significance of value creation in digital transformation according to the Chair of Information Systems at FAU?",
"How do institutional funding agreements or publisher memberships reduce or waive publication charges for authors?",
"What role do industry partnerships play in the Medical Engineering program at FAU?",
"How does FAU engage with the public through science communication?",
"What support does the library offer for using citation management tools such as Zotero or EndNote?",
"How does the Institute of Economic Research integrate mathematical and computational tools in economic modeling?"
"Welche internationalen Partnerschaften pflegt die FAU?",
"How does FAU ensure that theoretical foundations are linked to practical programming skills?",
"What challenges are faced in integrating AI systems into clinical workflows at FAU?",
"How does empirical educational research at FAU contribute to sustainable development and empowerment?",
"How is the state examination (Erste Juristische Prüfung) structured for law students at FAU?",
"How can students specialize within the Medical Engineering program at FAU?",
"How can SSD caching improve database or file throughput on a mirror server?",
"What interdisciplinary opportunities are available for students interested in theoretical computer science at FAU?",
"What are the legal rights authors grant when depositing a document in an institutional repository?",
"How do members of the Chair contribute to shaping academic debates in the areas of dynamics and numerics?"
"How do students at FAU elect their representatives?",
"How is religious education in schools structured through Protestant theology at FAU?",
"How is sustainability integrated into the curriculum at the School of Business, Economics and Society at FAU?",
"How does FAU assess the economic trade-offs in different energy transition pathways?",
"What are the main research areas within the Faculty of Engineering at FAU?",
"How do mirror servers report reliability, sync status, and exit codes of update jobs?",
"How do students in the FAU Medical Engineering program engage with current medical technologies?",
"Welche Herausforderungen ergeben sich bei der Analyse großer medizinischer Bilddatensätze?",
"What are the main research areas in the Department of Chemistry and Pharmacy at FAU?",
"Wie beteiligt sich die FAU an europäischen Forschungsprojekten?",
"What elective courses are available for specialization in the FAU Medical Engineering program?",
"What are the key research areas in the field of industrial engineering and management at FAU?",
"What are the steps to use library catalogs effectively at the FAU?",
"How does the Chair of Information Systems at FAU explore the impact of digital ecosystems on business?",
"How are mobile device management solutions implemented in academic environments?",
"How does the Chair of Information Systems at FAU approach teaching agile project management?",
"What are the most common issues addressed by student representation at FAU?",
"What career paths are available to graduates of the Artificial Intelligence programs at FAU?",
"What computational tools and frameworks are taught or used in the Machine Learning and Data Analytics Lab at FAU?",
"How is robotics taught within the Artificial Intelligence curriculum at FAU?",
"What is the structure and governance model of FAU?",
"What is the role of cloud-based manufacturing platforms in FAU’s research?",
"What is the role of academic-industry partnerships in the work of the Chair of Information Systems at FAU?",
"How does the Chair of Information Systems at FAU integrate real-world business challenges into its teaching?",
"How does FAU support innovation and entrepreneurship in electrical engineering?",
"What are the main research focus areas at the Faculty of Medicine at FAU?",
"How does FAU examine religious identity in multicultural societies?",
"How can students provide feedback to improve strategic management courses at FAU?",
"How does FAU support student involvement in conferences and academic publishing?",
"What academic backgrounds are best suited for success in the Artificial Intelligence programs at FAU?",
"How are students trained to think critically in strategic management courses at FAU?",
"How are optical and photonic systems studied at FAU?",
"How is medical education structured at the Faculty of Medicine at FAU?",
"What is the role of stakeholder analysis in the strategic management curriculum at FAU?",
"How do students benefit from the university’s location in the Nuremberg-Erlangen industrial region?",
"What is the balance between theory and practical implementation in the Artificial Intelligence programs at FAU?",
"What are the main challenges students face during their studies at the School of Business, Economics and Society at FAU, and how are they addressed?",
"How are proof assistants like Coq or Isabelle used in courses at FAU?",
"What are the main focus areas of the Artificial Intelligence study programs at FAU?",
"How do educational programs integrate high performance computing (HPC) training for students?",
"What role do metrics and analytics play in managing university IT services?"
"How does FAU support interdisciplinary projects between Artificial Intelligence and health sciences?",
"How does the Faculty of Law at FAU integrate international law into its curriculum?",
"What are the security considerations in high performance computing (HPC) infrastructures?",
"What kind of grading or credit recognition (e.g. ECTS credits) is associated with successful completion of study‑accompanying German courses?",
"What are the main challenges faced by students studying strategic management at FAU?",
"What are the learning outcomes expected from the strategic management courses at FAU?",
"What roles do HTTP, HTTPS and RSYNC protocols play in software mirror infrastructures?",
"What opportunities exist for interdisciplinary work between Artificial Intelligence and social sciences at FAU?",
"How does virtualization affect high performance computing (HPC) performance?",
"How is medical device regulation and safety covered in the FAU Medical Engineering program?",
"How is the German legal system introduced to international students studying at FAU?",
"Was zeichnet das Campusleben an der FAU aus?",
"What are the core responsibilities of a university computing center in supporting academic institutions?",
"What is the importance of Roman law in the legal education at FAU?",
"What mentoring or advising resources are available for electrical engineering students at FAU?",
"What roles do numerical analysis and computational mathematics play in solving partial differential equations at the Chair?",
"What strategies are used to reduce the environmental impact of university IT infrastructure?",
"How is IT infrastructure adapted to support international collaborations in research and education?",
"How can students propose changes to university policies through student representation at FAU?",
"What programming languages are used to demonstrate concepts in compiler construction at FAU?",
"What kind of facilities and equipment are available for Medical Engineering students at FAU?",
"How does the Department of German as a Foreign Language define and support different proficiency levels aligned with the Common European Framework of Reference for Languages (CEFR)?",
"What is the difference between tightly coupled and loosely coupled tasks in high performance computing (HPC)?",
"How does FAU promote diversity and inclusion within its electrical engineering programs?",
"What kind of partnerships does FAU have with companies for internships or thesis work in industrial engineering?",
"What interdisciplinary research is conducted by electrical engineering departments at FAU?",
"What dual degree or combined degree options are available at the Faculty of Medicine at FAU?",
"How does the FAU library support distance or online students?",
"How does the Institute of Economic Research study employment research topics at FAU?",
"How do student representatives at FAU advocate for student rights?",
"What programming languages or software tools are commonly used in the Medical Engineering program at FAU?",
"What are the main goals of the Medical Faculty’s internationalization strategy at FAU?"
"What are the formats and key components of the written and oral parts of the DSH examination?",
"What methodologies are used in theological research at FAU?",
"How does the Chair of Information Systems at FAU explore the role of culture in digital transformation?",
"How does the Institute of Economic Research investigate macroeconomics questions?",
"What role does logic play in the computer science curriculum at FAU?",
"What strategies are used to mirror high‑volume offline datasets like OpenStreetMap dumps?",
"What experience do students gain with medical sensors and instrumentation at FAU?",
"How are quantitative methods taught and applied in programs at the School of Business, Economics and Society at FAU?",
"Wie erfolgt die Integration von Bild‑ und Sprachdaten zur multimodalen Mustererkennung?",
"How are the study‑accompanying German courses structured in terms of learning objectives, content, and contact hours?",
"Welche Bedeutung haben Zeitreihenanalysen für Gesundheits‑ und Medizindaten im LME?",
"How does FAU incorporate critical theory into its study of religion?",
"What research opportunities exist for students in Artificial Intelligence at FAU?",
"What are the main research priorities of FAU?",
"What are the main research areas at the Institute of Economic Research at FAU?",
"What degree programs are offered by the Faculty of Law at FAU?",
"What types of Artificial Intelligence applications are students encouraged to explore at FAU?",
"What guiding principles shape the approach to empirical research at the Institute for Empirical Education and Social Science?"
"What are the best practices for collecting high-quality physiological data at FAU?",
"What is the relationship between formal logic and system security in the FAU curriculum?",
"What communication channels are used by student representatives to reach the wider student body at FAU?",
"What kind of feedback mechanisms exist for law students at FAU?",
"What ethical considerations are integrated into information systems research and teaching at FAU?",
"How is model checking introduced in computer science education at FAU?",
"What types of clinical conditions are modeled using machine learning at FAU?",
"What is the structure of the academic calendar at the School of Business, Economics and Society at FAU?",
"What interdisciplinary opportunities are there between business, economics, and sociology at the School of Business, Economics and Society at FAU?",
"How is mathematical control theory applied in the modeling and regulation of dynamic systems in research?",
"What are the primary applications of explainable AI in the healthcare domain at FAU?",
"How does FAU prepare students for careers in biomedical data science?",
"What are the main historical developments in Protestant theology covered at FAU?",
"How does the FAU Medical Engineering program address ethical issues in medical technology?",
"What research institutes are affiliated with the Faculty of Medicine at FAU?",
"How does the Faculty of Engineering integrate ethical considerations into its curriculum?",
"Wie werden inverse Probleme mit Anwendungen wie Bewegungsrekonstruktion adressiert?",
"What types of documents (such as dissertations, monographs, preprints, teaching materials) can be archived in an institutional open access system?",
"How are questions of religious pluralism addressed in religious studies at FAU?",
"How can students at FAU get involved in research during their Bachelor's degree?",
"How does the Artificial Intelligence curriculum at FAU compare to other leading universities?",
"How is human-machine collaboration optimized in production environments at FAU?",
"How is organic chemistry taught and researched within the department at FAU?",
"How does FAU support undergraduate and graduate research in programming language theory?",
"How can authors choose appropriate Creative Commons licences for their open access publications?",
"What are the key subject-specific collections available at the university library of FAU?",
"How does FAU encourage curiosity and exploration in abstract computer science domains?",
"Welche Schwerpunkte verfolgt das Labor in der Population Modelling Forschung?",
"What kinds of virtualization technologies are used in academic computing centers?",
"How does the department ensure safety in chemical laboratories at FAU?",
"What skills are emphasized in the education of future digital entrepreneurs at FAU?",
"How is doctoral education structured at FAU?",
"What is the process for writing a thesis in the law programs at FAU?",
"What kinds of thesis topics are typical in the chemistry Master’s program at FAU?",
"What are examples of innovations developed through the Medical Engineering program at FAU?",
"What are best practices for managing mirrored software archives for academic use?",
"What distinguishes FAU’s approach to teaching electrical engineering compared to other German universities?",
"How does electrical engineering education at FAU integrate theoretical and practical learning?",
"What interdisciplinary projects involve collaboration between the law faculty and other faculties at FAU?",
"Was sind die wichtigsten Ziele der Exzellenzstrategie der FAU?",
"How can students at FAU access archived or historical academic materials?",
"What are the ethical challenges explored in Artificial Intelligence courses at FAU?",
"How does the Institute of Economic Research collaborate with public finance scholars at FAU?",
"How is leadership in digital environments addressed in the curriculum at FAU?",
"How does the department balance classroom teaching with autonomous or media‑supported learning formats for German?",
"What project management techniques are taught in the industrial engineering and management program at FAU?",
"How are qualitative and quantitative methods balanced in research at the Chair of Information Systems at FAU?",
"What kinds of internships are recommended for law students at FAU?",
"How is symbolic logic applied in software correctness courses at FAU?",
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
