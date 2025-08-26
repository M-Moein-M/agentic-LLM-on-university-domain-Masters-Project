import json
import os
import asyncio

import requests
import random 
import concurrent
import aiohttp
import httpx
import time
from typing import List, Optional, Dict, Any, Union
from urllib.parse import unquote
import logging
#from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
from markdownify import markdownify
import aiohttp

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool

from langsmith import traceable

from elasticsearch import Elasticsearch

from open_deep_research.state import Section
from open_deep_research.configuration import Configuration, SearchAPI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from functools import lru_cache
from langchain_openai import ChatOpenAI   # pip install langchain-openai
from pathlib import Path
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    CrawlResult
)


logger = logging.getLogger(__name__)
import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

FAU_CORPUS = dict()
# with open("/home/horatio/projects/corpus/main_corpus_nhr_fau_de_eu.jsonl", "r", encoding="utf-8")as f:
with open("/home/horatio/projects/corpus/crawl4ai_corpus.jsonl", "r", encoding="utf-8")as f:
    for doc in f:
        doc = json.loads(doc)
        url = doc.get("url")
        if not url:
            url = doc["metadata"]["url"]
        if url:
            FAU_CORPUS.update({url: doc})
    print("Corpus loaded")

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    
    return datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%a %b %-d, %Y")


def get_config_value(value):
    """
    Helper function to handle string, dict, and enum cases of configuration values
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source=4000, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
            
    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str


@traceable
async def es_search(search_queries) -> list:
    """
    Search local documents using Elasticsearch
    returns the results and the surplus tokens
    """
    INCLUDE_TOP_N_RESULTS = 10
    MAX_TEXT_LENGTH = 10_000
    TOTAL_RESULT_BUDGET = 40_000*4  # character count
    search_docs = list()
    visited_urls = set()
    client = Elasticsearch(
        # For local development
        "http://localhost:9200",
        basic_auth=("elastic", "zb3BaJvO")
    )

    def get_results(query):
        """ send the request """

        res = client.search(
            index="search-cai",
            size=INCLUDE_TOP_N_RESULTS,
            query={
                "multi_match" : {
                    "query":    query,
                    "fields": ["text", "title", "description", "url"]
                }
            }
        )
        print(f"### got {len(res.body["hits"]["hits"])} hits from Elasticsearch")

        hits = list()
        for hit in res.body["hits"]["hits"]:
            if hit["_source"]["url"] in visited_urls:
                continue
            hits.append({
                "text": hit["_source"]["text"],
                "url": hit["_source"]["url"],
                "title": hit["_source"].get("title", "") #+ ";" +  hit["_source"].get("description", "")
            })
        
        
        return hits
    
    for query in search_queries:
        print(query)
        serp = get_results(query)
        # try with dockdockgo if google fails
        if len(serp) == 0:
            print("** NO RESULTS FOUND WITH ELASTICSEARCH **")

        results = list()
        i = 0
        for i, res in enumerate(serp):
            url = res["url"]
            if url in visited_urls or url.endswith(".pdf") or not res["text"] or len(res["text"]) > MAX_TEXT_LENGTH:
                continue
            visited_urls.add(url)
            res["raw_content"] = res["text"]  # to match searxng scheme
            results.append(res)
            if len(results) >= INCLUDE_TOP_N_RESULTS:
                break
        
        # Format response to match Tavily structure
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })

  
    return search_docs

@traceable
async def searxng_search(search_queries) -> (list, set):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    INCLUDE_TOP_N_RESULTS = 3
    MAX_TEXT_LENGTH = 9_000
    search_docs = []
    visited_urls = set()

    async def get_results(session, query, engine):
        """Send the request asynchronously."""
        domains_env = os.getenv("SEARXNG_DOMAINS", "fau.eu,fau.de")
        domains = [d.strip() for d in domains_env.split(",")]
        domains_str = ' OR '.join([f"site:{d}" for d in domains])
        
        url = f"http://localhost:8080/search?q={query} -filetype:pdf {domains_str}&format=json&engines={engine}"
        
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            return [r for r in data["results"] if r["url"] not in visited_urls]

    async with aiohttp.ClientSession() as session:
        for query in search_queries:
            print(query)
            serp = await get_results(session, query, "google")
            if len(serp) == 0:
                print("** using back up search engine")
                serp = await get_results(session, query, "duckduckgo")
            results = []
            all_urls = set([res["url"] for res in serp])

            for res in serp:
                url = res["url"]
                if url in visited_urls or url.endswith(".pdf"):
                    continue
                visited_urls.add(url)

                if FAU_CORPUS.get(url, None):
                    res["raw_content"] = FAU_CORPUS[url]["text"]
                    print("---- CACHE HIT:", url)
                else:
                    try:
                        crawl_result = await crawl_url_with_crawl4ai(url)
                        if crawl_result.success:
                            text = crawl_result.markdown.raw_markdown
                            FAU_CORPUS[url] = {"text": text}
                            res["raw_content"] = text
                            if text and len(text) > MAX_TEXT_LENGTH:
                                res["raw_content"] = text[:MAX_TEXT_LENGTH] + "... [truncated]"
                        else:
                            print("Error crawling url", crawl_result.error_message, url)
                    except Exception as e:
                        print("Error crawling url", e, url)

                results.append(res)
                if len(results) >= INCLUDE_TOP_N_RESULTS:
                    break

            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results
            })

    return search_docs, all_urls


# ---------------------------------------------------------------------
# Custom LLM bootstrapper
# ---------------------------------------------------------------------


@lru_cache(maxsize=4)
def _create_custom_chat_model(model_name: str) -> ChatOpenAI:
    """
    Return a ChatOpenAI that speaks to a self-hosted OpenAI-compatible
    endpoint.  Values are taken from either Configuration or env vars:
        CUSTOM_BASE_URL / BASE_URL   –  http://host:port/v1
        CUSTOM_API_URL               –  override completions URL
        CUSTOM_API_KEY               –  token if the gateway needs one
    """
    base_url = (
        os.getenv("CUSTOM_API_URL")            # most specific
        or os.getenv("CUSTOM_BASE_URL")
        or os.getenv("BASE_URL")
    )
    if not base_url:
        raise ValueError(
            "Set CUSTOM_BASE_URL (or BASE_URL) so we know where to talk to."
        )

    return ChatOpenAI(
        base_url="http://10.28.53.143:6000/v1", #base_url.rstrip("/"),
        api_key="xFhGltj52Gn",  # can be dummy
        model_name="/anvme/workspace/unrz103h-helma/base_models/full",
        temperature=0,
        streaming=True,
    )


@lru_cache(maxsize=4)
def _create_custom_chat_model(model_name: str, **common_kwargs) -> ChatOpenAI:
    base_url = (
        os.getenv("CUSTOM_API_URL")
        or os.getenv("CUSTOM_BASE_URL")
        or os.getenv("BASE_URL")
    )
    if not base_url:
        raise ValueError(
            "Set CUSTOM_BASE_URL (or BASE_URL) so the client knows where to connect"
        )

    # return ChatOpenAI(
    #     base_url=base_url.rstrip("/"),
    #     api_key="xFhGltj52Gn",  # ignored by many gateways
    #     model=model_name,
    #     **common_kwargs,
    # )
    return ChatOpenAI(
        base_url="http://10.28.53.143:6000/v1",#base_url.rstrip("/"),
        api_key="xFhGltj52Gn",  # can be dummy
        model_name="/anvme/workspace/unrz103h-helma/base_models/full",
        temperature=0,
        streaming=True,
    )


def init_chat_model(model: str, *, temperature: float = 0, streaming: bool = True, **kwargs):
    """Return a chat model for *model*.

    • ``custom:<name>``   – self‑hosted OpenAI‑compatible gateway 
    • ``groq:<name>``     – GroqCloud
    • ``ollama:<name>``   – local Ollama
    • anything else       – defaults to regular OpenAI
    """
    # ---------------------------  Custom  --------------------------
    return _create_custom_chat_model(
        "dummy",
        temperature=0,
        streaming=streaming,
        api_key="xFhGltj52Gn",        
        # **kwargs,
    )

def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.
    
    Iteratively removes all occurrences of content enclosed in thinking tokens.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        thinking_trace = text[start: end]
        text = text[:start] + text[end:]
    return text, thinking_trace

async def crawl_url_with_crawl4ai(url):
    browser_config = BrowserConfig(
        headless=True,
        verbose=True
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawler_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            )
        )
        result: CrawlResult = await crawler.arun(
            url=url,
            config=crawler_config,
            excluded_tags=['img', 'nav', 'header', 'footer', 'aside', 'a', 'href']
        )
        return result
