"""arXiv paper crawler."""
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from config import ARXIV_CATEGORIES, ARXIV_MAX_RESULTS_PER_QUERY

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search_papers(query: str = "", start: int = 0, max_results: int = None,
                  categories: list[str] = None) -> list[dict]:
    """Search arXiv for papers. Returns list of paper dicts."""
    max_results = max_results or ARXIV_MAX_RESULTS_PER_QUERY
    categories = categories or ARXIV_CATEGORIES

    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    if query:
        full_query = f"({query}) AND ({cat_query})"
    else:
        full_query = cat_query

    params = {
        "search_query": full_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_data = resp.read()
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        return []

    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")

        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.find("atom:name", NS)
            if name is not None:
                authors.append(name.text)

        cats = []
        for cat in entry.findall("arxiv:primary_category", NS):
            cats.append(cat.get("term"))
        for cat in entry.findall("atom:category", NS):
            t = cat.get("term")
            if t and t not in cats:
                cats.append(t)

        published = entry.find("atom:published", NS).text[:10]

        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": cats,
            "published_date": published,
            "pdf_url": pdf_url,
        })

    return papers


def fetch_recent(max_results: int = 100) -> list[dict]:
    """Fetch recent papers from the configured discovery profile."""
    return search_papers(max_results=max_results)
