"""Local, page-addressable vector retrieval; raw sources never become instructions."""
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_sources():
    return json.loads((ROOT / "data/sources.json").read_text())


def fetch_source(source):
    url = source.get("download", source["url"])
    response = requests.get(url, timeout=60, headers={"User-Agent": "KVCacheCourseResearch/1.0"})
    response.raise_for_status()
    if "/pdf/" in url or "application/pdf" in response.headers.get("content-type", ""):
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(response.content)).pages]
        pagination = "PDF page"
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        body = soup.find("main") or soup.find("article") or soup
        raw = body.get_text(" ", strip=True)
        if len(raw) < 150:
            raise ValueError(f"{source['id']}: source body too short")
        # Web documents have no pages: use an explicitly disclosed 3,000-character equivalent.
        pages = [raw[i:i + 3000] for i in range(0, len(raw), 3000)]
        pagination = "web equivalent (3000 characters)"
    if not pages or not any(p.strip() for p in pages):
        raise ValueError(f"{source['id']}: no extractable text")
    return {"source": source, "pages": pages, "pagination": pagination,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat()}


def prepare():
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for source in read_sources():
        path = directory / f"{source['id']}.json"
        record = json.loads(path.read_text()) if path.exists() else fetch_source(source)
        if record["source"] != source:
            raise ValueError(f"{source['id']}: cached manifest differs; remove its raw cache and prepare again")
        records.append(record)
        if sum(len(r["pages"]) for r in records) > 200:
            raise ValueError("RAG corpus exceeds 200 pages including web equivalents")
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"{source['id']}: {len(record['pages'])} pages ({record['pagination']})", flush=True)
    return records


class VectorRetriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import try_to_load_from_cache
        self.records = prepare()
        self.model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        cached = try_to_load_from_cache(self.model_name, "modules.json")
        self.model = SentenceTransformer(self.model_name, device="cpu", trust_remote_code=False,
                                        local_files_only=isinstance(cached, str))
        self.chunks = []
        for record in self.records:
            source = record["source"]
            for page, text in enumerate(record["pages"], 1):
                # Keep chunks below the E5 token limit, including PDF math-heavy pages.
                tokens = self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
                for offset in range(0, len(tokens), 240):
                    chunk = self.model.tokenizer.decode(tokens[offset:offset + 300], skip_special_tokens=True)
                    if chunk.strip():
                        self.chunks.append({"id": f"{source['id']}:p{page}:t{offset}", "source_id": source["id"],
                                            "tech": source["tech"], "scope": source["scope"],
                                            "page": page, "pagination": record["pagination"], "text": chunk})
        fingerprint = hashlib.sha256(json.dumps([self.model_name, self.chunks], sort_keys=True).encode()).hexdigest()
        directory = ROOT / "data/index"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{fingerprint}.npy"
        if path.exists():
            self.vectors = np.load(path, allow_pickle=False)
        else:
            self.vectors = self.model.encode(["passage: " + c["text"] for c in self.chunks],
                                            normalize_embeddings=True, show_progress_bar=True, batch_size=32)
            np.save(path, self.vectors)
        self.fingerprint = fingerprint

    def search(self, query, tech, source_ids=None, k=5):
        indices = [i for i, c in enumerate(self.chunks) if c["tech"] == tech and
                   (source_ids is None or c["source_id"] in source_ids)]
        if not indices:
            return []
        vector = self.model.encode("query: " + query, normalize_embeddings=True)
        scores = self.vectors[indices] @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [dict(self.chunks[indices[i]], similarity=round(float(scores[i]), 5)) for i in order]


def web_search(query):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live WebSearch")
    response = requests.post("https://api.openai.com/v1/responses", timeout=90,
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
                                   "store": False, "max_output_tokens": 450,
                                   "input": "Search the web for: " + query + ". Return a concise summary with source citations.",
                                   "tools": [{"type": "web_search", "search_context_size": "low"}],
                                   "include": ["web_search_call.action.sources"],
                                   "tool_choice": "required"})
    if not response.ok:
        raise RuntimeError(f"OpenAI WebSearch HTTP {response.status_code}")
    body = response.json()
    if body.get("status") != "completed" or not any(x.get("type") == "web_search_call" for x in body.get("output", [])):
        raise RuntimeError("OpenAI WebSearch did not complete a search call")
    results = []
    for item in body["output"]:
        for source in (item.get("action") or {}).get("sources", []):
            if source.get("type") == "url" and source.get("url"):
                result = {"title": source.get("title", ""), "url": source["url"], "snippet": ""}
                if result["url"] not in {r["url"] for r in results}:
                    results.append(result)
        for content in item.get("content", []):
            for annotation in content.get("annotations", []):
                if annotation.get("type") == "url_citation" and annotation.get("url"):
                    result = {"title": annotation.get("title", ""), "url": annotation["url"], "snippet": ""}
                    if result["url"] not in {r["url"] for r in results}:
                        results.append(result)
    if not results:
        raise RuntimeError("WebSearch returned no results; check search connectivity")
    return results[:5]
