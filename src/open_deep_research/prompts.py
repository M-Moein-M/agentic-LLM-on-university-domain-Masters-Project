from open_deep_research.utils import (
    get_today_str,
)


fast_answer_evaluator = """You are an expert in Quality evaluation of a retreival system
Based on the discussion obove, with having:
1. A set of **queries** related to FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg) setting.
2. A **text** that provides answers or context for those queries.

Your task is to evaluate the response based on the queries. Does the reponse provide enough context and if the context answers the queries? Only a boolean value is needed.
Evaluate if the answer to the qeuries is relevant or not by a True or False output.
"""

followup_seed_prompt = """
You are given:
1. A set of initial queries related to a university setting.
2. A text that provides answers to those queries.

Your task is to generate **new, thoughtful follow-up questions** that:
- Are inspired by the content and implications of the text.
- Go **beyond the scope** of the original queries to explore deeper, broader, or adjacent topics.
- Are phrased in **natural, human-like language**—not keyword fragments or search queries.
- Stay relevant to the domain of universities, such as research, labs, infrastructure, student experience, administration, policy, or education.

Guidelines:
- Don't repeat or rephrase the original queries.
- *IMPORTANT* Don't generate questions that the answer is already in the text. We want to explore new topics.
- Use the text as a foundation: What new questions does this answer raise?
- You may explore implications, causes, challenges, applications, or related domains.
- Aim for intellectual curiosity and progressive exploration.

Input format:
Queries:
{queries}
...

Text:
\"\"\"
{text}
\"\"\"
"""

fast_answer_system_prompt = """
You are an intelligent AI assistant who answers questions regarding FAU university.
Use the retriever tool available to answer questions about the FAU and any topic that can be related to this university. You can make multiple calls if needed.
If the question is not related to FAU and a general knowledge, answer it without accessing FAU tool.

When providing answer based on FAU retriever tool:
- IMPORTANT: Use tables and bullet points to make the answer more readable
- Add the exact sources/URLs for each section of the final answer so the user can cross check that
- Stick to the FAU context provided to you
- Avoid repeating ovelapping parts. Merge them to increase readability
- Keep things up to date if possible. Today is {today}.

""".format(today=get_today_str())

fast_query_writer="""You are performing research to provide material to answer a topic. 

<topic>
{topic}
</topic>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information to address the topic. Stick to the topic language to generate queries.

The queries should:

The search will be limited on resources from FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg). Focus on balancing breadth and depth of queries and do not mention references to FAU.
The topic can be a question or query from the user. Your task is to find the best SERP queries to address the topic.
Queries should look for most recent data. Today is {today}.
Feel free to output less queries if the topic is specific enough. Drop the similar queris and avoid overlapping.
</Task>

"""
