# path: asx_scraper.py

"""
ASXScraper with embedded hybrid AI document parser (pdfplumber -> OCR -> OpenAI)
Requirements:
  pip install openai pdfplumber pdf2image pytesseract PyPDF2 python-dotenv
  Install tesseract and poppler for OCR/pdf2image.
Set OPENAI_API_KEY in environment or .env (do NOT hardcode keys).
"""

import os
import re
import time
import json
import hashlib
import sqlite3
import pdfplumber
import pytesseract
import traceback
from pdf2image import convert_from_path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from contextlib import contextmanager
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import PyPDF2
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv()

# Lazy import openai to avoid hard crash when not installed
try:
    import openai
except Exception:
    openai = None

class ASXScraper:
    """
    ASX announcements scraper with hybrid AI PDF parsing.
    AI parsing always used (per user request).
    """

    def __init__(self, db_path: str = "asx_announcements.db"):
        self.base_url = "https://www.asx.com.au"
        self.download_dir = "./asx_downloads"
        self.announcements_page_url = "https://www.asx.com.au/markets/trade-our-cash-market/announcements"
        # Optional JSON API endpoint; if not set we try sensible defaults
        self.announcements_api_url = os.getenv("ASX_ANNOUNCEMENTS_API_URL")
        # Config: days back to search, only include targets, debug logging
        self.days_back = max(1, min(int(os.getenv("ASX_DAYS_BACK", "7") or 7), 60))
        def _parse_bool(val: Optional[str]) -> bool:
            if val is None:
                return False
            v = val.strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        # default behavior is to only return target announcements
        self.only_targets = True
        # If ASX_INCLUDE_ALL is true, include all
        if _parse_bool(os.getenv("ASX_INCLUDE_ALL")):
            self.only_targets = False
        # If ASX_ONLY_TARGETS is explicitly provided, respect it
        if os.getenv("ASX_ONLY_TARGETS") is not None:
            self.only_targets = _parse_bool(os.getenv("ASX_ONLY_TARGETS"))
        self.debug_fetch = _parse_bool(os.getenv("ASX_DEBUG_FETCH", "false"))
        self.match_regex_override = os.getenv("ASX_MATCH_REGEX")
        self.extra_keywords = [k.strip().lower() for k in (os.getenv("ASX_KEYWORDS", "")) .split(",") if k.strip()] if os.getenv("ASX_KEYWORDS") else []
        self.db_path = db_path

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'en-US,en;q=0.5'
        })

        # OpenAI API key lookup with multiple fallbacks
        # Primary: OPENAI_API_KEY; also support OPEN_API_KEY and common aliases
        self.openai_api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_TOKEN")
        )
        if not self.openai_api_key:
            # As a convenience, if a file named 'OPEN_API_KEY' exists, read from it
            key_file_path = os.path.join(os.getcwd(), "OPEN_API_KEY")
            if os.path.isfile(key_file_path):
                try:
                    with open(key_file_path, "r", encoding="utf-8") as f:
                        key_candidate = f.read().strip()
                        if key_candidate:
                            self.openai_api_key = key_candidate
                except Exception:
                    pass

        self._use_new_openai = False
        self._openai_client = None
        self._OpenAI_cls = None

        if not self.openai_api_key:
            print("Warning: OpenAI API key not set. Set OPENAI_API_KEY or OPEN_API_KEY in your environment or .env.")
        else:
            try:
                from openai import OpenAI  # type: ignore
                self._OpenAI_cls = OpenAI
            except Exception:
                self._OpenAI_cls = None

            if self._OpenAI_cls:
                try:
                    self._openai_client = self._OpenAI_cls(api_key=self.openai_api_key)
                    self._use_new_openai = True
                except Exception:
                    self._openai_client = None
                    self._use_new_openai = False

            if not self._use_new_openai and openai:
                openai.api_key = self.openai_api_key

        self.init_database()

    # ---------- Database helpers ----------
    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='announcements'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS announcements (
                        id TEXT PRIMARY KEY,
                        asx_code TEXT NOT NULL,
                        company_name TEXT,
                        document_type TEXT NOT NULL,
                        document_title TEXT NOT NULL,
                        announcement_date TEXT NOT NULL,
                        submission_date TEXT,
                        pdf_url TEXT,
                        transaction_type TEXT,
                        shares_amount INTEGER,
                        is_significant BOOLEAN,
                        significance_percentage REAL,
                        director_name TEXT,
                        processed BOOLEAN DEFAULT 0,
                        processing_error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(asx_code, announcement_date, document_title)
                    )
                ''')

            # migrations: ensure some columns exist (safe noop if already present)
            try:
                cursor.execute("SELECT director_name FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN director_name TEXT")
            try:
                cursor.execute("SELECT transaction_type FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN transaction_type TEXT")
            try:
                cursor.execute("SELECT shares_amount FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN shares_amount INTEGER")

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_asx_code ON announcements(asx_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_significant ON announcements(is_significant)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON announcements(announcement_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_type ON announcements(document_type)')
            conn.commit()

    # ---------- Utility ----------
    def generate_unique_id(self, asx_code: str, document_title: str, date: str) -> str:
        unique_string = f"{asx_code}_{document_title}_{date}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    # ---------- Fetching announcements (JSON API with HTML fallback) ----------
    def fetch_announcements(self, limit: int = 20, source: str = "json") -> List[Dict]:
        source = (source or "json").lower()
        if source not in ("json", "html", "auto"):
            source = "json"

        if source in ("json", "auto"):
            anns = self._fetch_announcements_markit(limit=limit)
            if anns:
                print(f"[debug] Markit JSON feed returned {len(anns)} announcements (limit {limit}).")
                return anns[:limit]
            if source == "json":
                print("[info] JSON feed returned no announcements; falling back to HTML.")

        html_anns = self._fetch_announcements_html(limit=limit)
        print(f"[debug] HTML scrape returned {len(html_anns)} announcements (limit {limit}).")
        return html_anns

    def _fetch_announcements_markit(self, limit: int = 20) -> List[Dict]:
        """
        Fetch announcements via Markit Digital JSON feed.
        """
        url = "https://asx.api.markitdigital.com/asx-research/1.0/home/announcements"
        params = {
            "numberOfAnnouncements": str(max(1, min(limit, 50))),
        }
        try:
            resp = self.session.get(url, params=params, timeout=20)
        except Exception as exc:
            print(f"[debug] JSON feed error: {exc}")
            return []

        if resp.status_code != 200:
            print(f"[debug] JSON feed status {resp.status_code}")
            return []

        try:
            payload = resp.json()
        except ValueError:
            print("[debug] JSON feed returned invalid JSON.")
            return []

        items = []
        data_block = payload.get("data")
        if isinstance(data_block, dict):
            items = data_block.get("items") or []
        if not isinstance(items, list):
            print("[debug] JSON feed items missing or malformed.")
            return []

        announcements: List[Dict] = []
        for entry in items:
            if len(announcements) >= limit:
                break

            try:
                code = (entry.get("symbol") or "").strip()
                headline = (entry.get("headline") or "").strip()
                document_key = (entry.get("documentKey") or "").strip()
                iso_date = entry.get("date") or ""
                company_name = (entry.get("displayName") or "").strip()

                if not code or not headline or not document_key:
                    continue

                announcement_date = ""
                if isinstance(iso_date, str) and iso_date:
                    try:
                        announcement_date = datetime.fromisoformat(iso_date.replace("Z", "+00:00")).date().isoformat()
                    except Exception:
                        announcement_date = iso_date

                pdf_url = f"https://cdn-api.markitdigital.com/apiman-gateway/ASX/asx-research/1.0/file/{document_key}"
                announcement_id = document_key

                announcements.append({
                    "code": code,
                    "headline": headline,
                    "date": announcement_date,
                    "id": announcement_id,
                    "url": pdf_url,
                    "company_name": company_name,
                })
            except Exception:
                continue

        return announcements

    def _fetch_announcements_json(self, limit: int = 20) -> List[Dict]:
        """
        Fetch announcements via ASX JSON API. If ASX_ANNOUNCEMENTS_API_URL is set, use it.
        Otherwise, try a small set of known endpoints. Returns a list of dicts
        with keys: code, headline, date (YYYY-MM-DD or source format), id, url.
        """
        announcements: List[Dict] = []
        # Candidate API endpoints to try (first configured via env)
        candidate_urls: List[str] = []
        if self.announcements_api_url:
            candidate_urls.append(self.announcements_api_url)
        # Common public endpoints historically used by ASX (may change)
        candidate_urls.extend([
            "https://www.asx.com.au/asx/1/announcements",
            # Some deployments expose a search endpoint; harmless to try
            "https://www.asx.com.au/asx/1/announcements/search",
        ])

        # Build a reasonable time window (last 7 days) to avoid empty results
        # Use timezone-aware UTC datetimes to avoid deprecation warnings
        now_utc = datetime.now(timezone.utc)

        for base in candidate_urls:
            try:
                # Try up to the last 7 days until we accumulate enough items
                for day_offset in range(0, max(1, self.days_back)):
                    if len(announcements) >= limit:
                        break
                    date_iso = (now_utc - timedelta(days=day_offset)).date().isoformat()
                    # Try common param names; some endpoints ignore unknown params
                    params = {
                        "limit": str(max(1, min(limit, 100))),
                        "itemsPerPage": str(max(1, min(limit, 100))),
                        "publishedAfter": f"{date_iso}T00:00:00Z",
                        "publishedBefore": f"{date_iso}T23:59:59Z",
                    }
                    print(f"[debug] JSON fetch {base} date={date_iso}")
                    resp = self.session.get(base, params=params, timeout=15)
                    if resp.status_code != 200:
                        print(f"[debug]  -> status {resp.status_code}, skipping.")
                        continue
                    data = resp.json()
                    print(f"[debug]  -> received JSON type {type(data).__name__}")

                    # Normalize items array from various possible shapes
                    items = None
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        for key in ("data", "items", "results", "announcements", "content"):
                            if key in data and isinstance(data[key], list):
                                items = data[key]
                                break
                        # Some APIs nest a page object
                        if items is None:
                            for key in ("page", "payload"):
                                if key in data and isinstance(data[key], dict):
                                    nested = data[key]
                                    for k in ("items", "content", "data"):
                                        if k in nested and isinstance(nested[k], list):
                                            items = nested[k]
                                            break
                                    if items is not None:
                                        break

                    if not items:
                        continue

                for it in items:
                    try:
                        code = (it.get("code") or it.get("issuerCode") or it.get("securityCode") or
                                it.get("companyCode") or it.get("asxCode") or "").strip()
                        headline = (it.get("headline") or it.get("title") or it.get("documentHeadline") or
                                    it.get("announcementTitle") or "").strip()

                            # Dates frequently come as ISO timestamps
                        raw_date = (it.get("date") or it.get("publishedAt") or it.get("announcementDate") or
                                    it.get("time") or it.get("published") or "")
                        date_str = ""
                        if isinstance(raw_date, str) and raw_date:
                            # Try ISO date extraction
                            m = re.match(r"(\d{4}-\d{2}-\d{2})", raw_date)
                            if m:
                                date_str = m.group(1)
                            else:
                                # Try dd/mm/yyyy
                                m2 = re.match(r"(\d{2}/\d{2}/\d{4})", raw_date)
                                date_str = m2.group(1) if m2 else raw_date

                            # Extract idsId or similar id to build the PDF URL
                        ids_id = (it.get("idsId") or it.get("id") or it.get("documentId") or
                                  it.get("announcementId") or "")

                        pdf_candidate = None
                        if isinstance(ids_id, (str, int)) and str(ids_id):
                            pdf_candidate = (
                                f"https://www.asx.com.au/asx/statistics/displayAnnouncement.do?display=pdf&idsId={ids_id}"
                            )
                        else:
                            pdf_candidate = it.get("pdfUrl") or it.get("url") or None

                        pdf_url = self._resolve_pdf_url(pdf_candidate)

                        if not code or not headline or not pdf_url:
                            if self.debug_fetch:
                                print(f"[debug]  -> skip item code={code} headline={headline} pdf_url={pdf_url}")
                            continue

                        ann_id = ""
                        if isinstance(ids_id, (str, int)) and str(ids_id):
                            ann_id = str(ids_id)
                        else:
                            ann_id = self._extract_ids_id(pdf_candidate) if pdf_candidate else ""
                        if not ann_id:
                            ann_id = self.generate_unique_id(code, headline, date_str or raw_date or str(time.time()))

                        ann = {
                            "code": code,
                            "headline": headline,
                            "date": date_str,
                            "id": ann_id,
                            "url": pdf_url,
                        }
                        if self.only_targets:
                            if self.is_target_announcement(headline):
                                announcements.append(ann)
                                if self.debug_fetch:
                                    print(f"[ASX JSON] match: {code} | {headline}")
                        else:
                            announcements.append(ann)
                            if self.debug_fetch:
                                print(f"[ASX JSON] all: {code} | {headline}")
                        if len(announcements) >= limit:
                            break
                    except Exception:
                        continue

                    if len(announcements) >= limit:
                        break

                if announcements:
                    return announcements
            except Exception:
                continue

        return announcements

    def _fetch_announcements_html(self, limit: int = 20) -> List[Dict]:
        """Fallback HTML scraper if JSON API is unavailable."""
        try:
            url = "https://www.asx.com.au/asx/v2/statistics/todayAnns.do"
            resp = self.session.get(url, timeout=15)
            print(f"[debug] HTML fetch {url} status={resp.status_code}")
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find("table")
            if not table:
                print("[debug] No <table> found in HTML response.")
                return []
            rows = table.find_all("tr")
            announcements: List[Dict] = []
            print(f"[debug] Parsed {len(rows)} table rows from HTML.")
            if not rows or len(rows) <= 1:
                return announcements
            for row in rows:
                headers = row.find_all("th")
                if headers:
                    continue  # skip header row
                if len(announcements) >= limit:
                    break
                try:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        code = cells[0].get_text(strip=True)
                        date_cell = cells[1].get_text(" ", strip=True)
                        date_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", date_cell)
                        date_str = date_match.group(1) if date_match else date_cell
                        anchor_cell = cells[-1] if cells else None
                        headline_cell = anchor_cell.find('a') if anchor_cell else None
                        headline = ""
                        if headline_cell:
                            headline_parts = list(headline_cell.stripped_strings)
                            headline = headline_parts[0] if headline_parts else ""
                        doc_link = headline_cell.get('href') if headline_cell else ""
                        pdf_url = self._resolve_pdf_url(doc_link)
                        if self.debug_fetch:
                            print(f"[debug] HTML row code={code} date={date_str} headline={headline} link={doc_link} pdf={pdf_url}")
                        if not pdf_url:
                            continue
                        doc_id = self._extract_ids_id(doc_link) or self._extract_ids_id(pdf_url)
                        if not doc_id:
                            doc_id = self.generate_unique_id(code, headline, date_str or str(time.time()))
                        ann = {'code': code, 'headline': headline, 'date': date_str, 'id': doc_id, 'url': pdf_url}
                        if self.only_targets:
                            if self.is_target_announcement(headline):
                                announcements.append(ann)
                                if self.debug_fetch:
                                    print(f"[ASX HTML] match: {code} | {headline}")
                        else:
                            announcements.append(ann)
                            if self.debug_fetch:
                                print(f"[ASX HTML] all: {code} | {headline}")
                except Exception:
                    continue
            return announcements
        except Exception:
            return []

    def is_target_announcement(self, headline: str) -> bool:
        h = (headline or "").strip().lower()
        # Normalize common punctuation variants
        h = h.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")
        if self.match_regex_override:
            try:
                return re.search(self.match_regex_override, h, re.I) is not None
            except Exception:
                pass

        # Regex patterns capturing common variations
        patterns = [
            r"appendix\s*3x",
            r"appendix\s*3y",
            r"appendix\s*3z",
            r"change\s+of\s+director",
            r"director[^\n]*interest",
            r"form\s*603",
            r"form\s*604",
            r"form\s*605",
            r"substantial\s+(holder|shareholder|holding)",
            r"initial\s+director",
            r"final\s+director",
        ]
        # Keyword fallback including user-provided extras
        keywords = [
            'appendix 3y', "change of director", "director's interest",
            'appendix 3x', 'appendix 3z', 'initial director', 'final director',
            'form 603', 'form 604', 'form 605',
            'substantial holder', 'substantial shareholder', 'substantial holding'
        ] + self.extra_keywords

        if any(re.search(p, h, re.I) for p in patterns):
            return True
        return any(k in h for k in keywords)

    def determine_document_type(self, headline: str) -> str:
        """
        Heuristically determine document type from headline text.
        """
        h = (headline or "").lower()
        if "appendix 3x" in h or "initial director" in h:
            return "APPENDIX_3X"
        if "appendix 3y" in h or "change of director" in h or "director's interest" in h:
            return "APPENDIX_3Y"
        if "appendix 3z" in h or "final director" in h:
            return "APPENDIX_3Y"  # Treat 3Z as director interest class for formatting
        if ("form 603" in h or "form 604" in h or "form 605" in h or
                "substantial holder" in h or "substantial shareholder" in h or "substantial holding" in h):
            return "FORM_604"  # Normalize 603/605 to 604 family for downstream formatting
        return "OTHER"

    def _chat(self, **kwargs):
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")

        if self._use_new_openai and self._openai_client:
            return self._openai_client.chat.completions.create(**kwargs)

        if openai:
            return openai.ChatCompletion.create(**kwargs)

        raise RuntimeError("OpenAI SDK not available")

    def _resolve_pdf_url(self, href: Optional[str]) -> Optional[str]:
        if not href:
            return None

        absolute = urljoin(self.base_url, href)

        if absolute.startswith("//"):
            absolute = f"https:{absolute}"

        if "announcements.asx.com.au/asxpdf" in absolute.lower():
            return absolute

        try:
            resp = self.session.get(absolute, timeout=30)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        match = re.search(r"https://announcements\.asx\.com\.au/asxpdf/[^\"']+\.pdf", resp.text)
        if match:
            return match.group(0)

        return None

    @staticmethod
    def _extract_ids_id(href: Optional[str]) -> str:
        if not href:
            return ""
        match = re.search(r"idsId=([0-9A-Za-z]+)", href)
        return match.group(1) if match else ""

    # ---------- Downloads & text extraction ----------
    def download_pdf(self, url: str) -> str:
        if not url:
            return ""
        filepath = ""
        try:
            timestamp = int(time.time())
            filename = f"asx_document_{timestamp}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            resp = self.session.get(url, timeout=30, allow_redirects=True, stream=True)
            if resp.status_code != 200:
                return ""

            content_type = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return ""

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            return filepath
        except Exception:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        # Primary extraction with pdfplumber, fallback to PyPDF2 minimal pages
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts).strip()
        except Exception:
            try:
                text = ""
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for i in range(min(5, len(reader.pages))):
                        ptext = reader.pages[i].extract_text()
                        if ptext:
                            text += ptext + "\n"
                return text.strip()
            except Exception:
                return ""

    def _ocr_pdf_to_text(self, pdf_path: str) -> str:
        # Convert PDF pages to images and run pytesseract
        try:
            images = convert_from_path(pdf_path)
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_text).strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    # ---------- AI parsing ----------
    def _normalize_text(self, text: str) -> str:
        # Collapse multiple whitespace; keep newlines for context
        if not text:
            return ""
        # replace weird non-breaking etc with space
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\r\n', '\n', text)
        # collapse many spaces but preserve paragraph breaks
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 6000) -> List[str]:
        # chunk by characters, try to split on newlines near boundary
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            # try to move end backward to last newline for coherence
            if end < n:
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl
            chunks.append(text[start:end].strip())
            start = end
        return [c for c in chunks if c]

    def parse_pdf_with_ai(self, pdf_path: str, doc_title: Optional[str] = None) -> Dict:
        """
        Hybrid parse: extract text (pdfplumber -> OCR), normalize, chunk, call OpenAI,
        and return structured dict. Always used.
        """
        result = {
            "company_name": None,
            "director_name": None,
            "document_type": None,
            "transaction_type": None,
            "shares_acquired": None,
            "shares_disposed": None,
            "shares_before": None,
            "shares_after": None,
            "date_of_change": None,
            "consideration": None,
            "nature_of_change": None,
            "raw_ai": None,
            "error": None
        }

        # Step 1: extract text
        text = self.extract_text_from_pdf(pdf_path)
        if len((text or "").strip()) < 200:
            # fallback to OCR
            ocr_text = self._ocr_pdf_to_text(pdf_path)
            text = (ocr_text or text or "")
        text = self._normalize_text(text)
        if not text:
            result["error"] = "No text extracted from PDF (even after OCR)."
            return result

        # Step 2: ensure openai available
        if not openai or not self.openai_api_key:
            result["error"] = "OpenAI library or API key not available. Set OPENAI_API_KEY."
            return result

        # Compose instruction prompt for structured extraction
        system_prompt = (
            "You are an expert at reading Australian Securities Exchange (ASX) announcements "
            "and extracting structured data in JSON. Return JSON only, parsable by a machine. "
            "If a field is not present, set it to null or empty string as appropriate.\n"
            "Fields to extract:\n"
            "- company_name (string)\n"
            "- document_type (one of: APPENDIX_3X, APPENDIX_3Y, FORM_604, OTHER)\n"
            "- director_name (string)\n"
            "- transaction_type (buy/sell/issue/initial/substantial_change/other/null)\n"
            "- shares_acquired (integer or null)\n"
            "- shares_disposed (integer or null)\n"
            "- shares_before (integer or null)\n"
            "- shares_after (integer or null)\n"
            "- date_of_change (YYYY-MM-DD or original format string or null)\n"
            "- consideration (string or null)\n"
            "- nature_of_change (string or null)\n"
            "Also include a short 'confidence' score between 0 and 1 as 'confidence' and include 'notes' if needed."
        )

        # We will send chunks and ask the model to respond with JSON for each chunk, then merge.
        chunks = self._chunk_text(text, chunk_size=6000)
        aggregated = {}
        aggregation_notes = []
        model_responses = []
        try:
            for i, chunk in enumerate(chunks):
                prompt = (
                    f"{system_prompt}\n\n"
                    f"Document title (if available): {doc_title or ''}\n"
                    f"Document chunk {i+1}/{len(chunks)}:\n\n{chunk}\n\n"
                    "Return JSON only."
                )

                # Use chat completion for structured output
                # Model name may be changed by you; using gpt-4o-mini as suggested
                resp = self._chat(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.0,
                    n=1
                )
                content = (resp.choices[0].message.content or "").strip()
                model_responses.append(content)

                # parse JSON; if parsing fails, keep raw text
                try:
                    parsed = json.loads(content)
                except Exception:
                    # try to extract first JSON-like block in content
                    m = re.search(r'(\{.*\})', content, re.S)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                        except Exception:
                            parsed = {"raw": content}
                    else:
                        parsed = {"raw": content}

                # merge: prefer non-null fields, take numeric max for shares (if multiple)
                for k, v in parsed.items():
                    if v is None or v == "":
                        continue
                    # numeric merging rules
                    if k in ("shares_acquired", "shares_disposed", "shares_before", "shares_after"):
                        try:
                            iv = int(v)
                            prev = aggregated.get(k)
                            if prev is None:
                                aggregated[k] = iv
                            else:
                                # prefer the larger absolute numeric value as likely complete
                                if abs(iv) > abs(prev):
                                    aggregated[k] = iv
                        except Exception:
                            aggregated[k] = aggregated.get(k) or v
                    else:
                        # for other fields, set if not already set
                        if k not in aggregated or not aggregated[k]:
                            aggregated[k] = v
                aggregation_notes.append(f"chunk_{i+1}_len={len(chunk)}")
            # place aggregated into result
            for k in result.keys():
                if k in aggregated:
                    result[k] = aggregated[k]
            result["raw_ai"] = model_responses
            result["notes"] = "; ".join(aggregation_notes)
            # Post-process certain fields
            # Normalize document_type to known codes
            dt = result.get("document_type")
            if isinstance(dt, str):
                up = dt.upper()
                if "3X" in up:
                    result["document_type"] = "APPENDIX_3X"
                elif "3Y" in up:
                    result["document_type"] = "APPENDIX_3Y"
                elif "604" in up or "FORM 604" in up or "SUBSTANTIAL" in up:
                    result["document_type"] = "FORM_604"
                else:
                    result["document_type"] = "OTHER"
            # Normalize date format if possible
            doh = result.get("date_of_change")
            if doh:
                # attempt parse common formats
                for fmt in ("%d %B %Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
                    try:
                        dtobj = datetime.strptime(doh, fmt)
                        result["date_of_change"] = dtobj.strftime("%Y-%m-%d")
                        break
                    except Exception:
                        continue
            return result
        except Exception as e:
            tb = traceback.format_exc()
            result["error"] = f"AI parsing error: {e}\n{tb}"
            return result

    # ---------- Analysis formatters (retain existing formats) ----------
    def format_analysis_output(self, data: Dict) -> str:
        out = []
        if data.get('company_name'):
            out.append(f"**Australian Company:** {data.get('company_name')}")
        if data.get('date_of_change'):
            out.append(f"**Date:** {data.get('date_of_change')}")
        # shares
        sa = data.get('shares_acquired') or 0
        sd = data.get('shares_disposed') or 0
        net = (sa - sd) if (isinstance(sa, int) and isinstance(sd, int)) else None
        out.append("**Number of Shares:**")
        out.append(f"- **Acquired:** {sa:,}" if isinstance(sa, int) else f"- **Acquired:** {sa}")
        out.append(f"- **Disposed:** {sd:,}" if isinstance(sd, int) else f"- **Disposed:** {sd}")
        if net is not None:
            ct = "GAINED" if net > 0 else "LOST" if net < 0 else "NO CHANGE"
            out.append(f"- **Net change:** {abs(net):,} ({ct})")
        if data.get('director_name'):
            out.append(f"\nAdditional details:\n- The director is {data.get('director_name')}")
        if data.get('nature_of_change'):
            out.append(f"- {data.get('nature_of_change')}")
        if data.get('consideration'):
            out.append(f"- Consideration: {data.get('consideration')}")
        return "\n".join(out)

    def format_appendix_3x_output(self, data: Dict) -> str:
        out = []
        if data.get('company_name'):
            out.append(f"**Australian Company:** {data.get('company_name')}")
        if data.get('date_of_change'):
            out.append(f"**Date:** Initial notice dated {data.get('date_of_change')}")
        out.append("\n**Notice Type:** Initial Director's Interest Notice (Appendix 3X)")
        if data.get('director_name'):
            out.append(f"**Director:** {data.get('director_name')}")
        if data.get('shares_after'):
            out.append(f"**Initial Holding:** {data.get('shares_after'):,} ordinary shares")
        if data.get('nature_of_change'):
            out.append(f"\n**Nature of Interest:** {data.get('nature_of_change')}")
        return "\n".join(out)

    def format_form_604_output(self, data: Dict) -> str:
        out = []
        if data.get('company_name'):
            out.append(f"**Australian Company:** {data.get('company_name')}")
        if data.get('date_of_change'):
            out.append(f"**Date:** Notice dated {data.get('date_of_change')}")
        out.append("\n**Notice Type:** Substantial Shareholder Notice (Form 604)")
        if data.get('director_name'):
            out.append(f"**Substantial Holder:** {data.get('director_name')}")
        if data.get('shares_before') is not None:
            out.append(f"**Previous Holding:** {data.get('shares_before')}")
        if data.get('shares_after') is not None:
            out.append(f"**Current Holding:** {data.get('shares_after')}")
        return "\n".join(out)

    # ---------- Main processing flow (uses AI always) ----------
    def process_announcement(self, announcement: Dict) -> bool:
        try:
            asx_code = announcement.get('code', '')
            if not asx_code:
                return False

            document_title = announcement.get('headline', '')
            announcement_date = announcement.get('date', '')
            if not document_title or not announcement_date:
                return False

            # normalize date string
            try:
                # try ISO-like
                date_obj = datetime.strptime(announcement_date, '%Y-%m-%d')
                announcement_date = date_obj.strftime('%Y-%m-%d')
            except Exception:
                try:
                    date_obj = datetime.strptime(announcement_date, '%d/%m/%Y')
                    announcement_date = date_obj.strftime('%Y-%m-%d')
                except Exception:
                    pass  # keep as-is

            announcement_id = self.generate_unique_id(asx_code, document_title, announcement_date)
            company_name = announcement.get('company_name', asx_code)
            document_type = self.determine_document_type(document_title)
            pdf_url = announcement.get('url', None)

            # default placeholders
            transaction_type = ''
            shares_amount = 0
            director_name = ''
            analysis_output = ''

            # If there's a PDF, download it
            pdf_path = None
            if pdf_url:
                pdf_path = self.download_pdf(pdf_url)
            else:
                # no PDF — still attempt to call AI with the headline + metadata (very limited)
                # create a small temporary text file for AI to parse
                pdf_path = None

            # Always use AI parser as requested
            ai_result = None
            if pdf_path:
                ai_result = self.parse_pdf_with_ai(pdf_path, doc_title=document_title)
            else:
                # If no PDF, we can call the model with the headline only (best-effort)
                if self.openai_api_key:
                    try:
                        prompt = (
                            "You are given an ASX announcement headline. Try to infer document_type and "
                            "basic fields. Return JSON only with the same fields as full parser.\n\n"
                            f"Headline: {document_title}"
                        )
                        resp = self._chat(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=500
                        )
                        content = (resp.choices[0].message.content or "").strip()
                        try:
                            ai_result = json.loads(content)
                        except Exception:
                            ai_result = {"raw": content}
                    except Exception as e:
                        ai_result = {"error": f"AI headline parse failed: {e}"}
                else:
                    ai_result = {"error": "No PDF and OpenAI not configured."}

            # If AI returned anything useful, map to fields
            if ai_result:
                # Use fields from AI when present otherwise fallback
                company_name = ai_result.get('company_name') or company_name
                director_name = ai_result.get('director_name') or director_name
                document_type_ai = ai_result.get('document_type') or document_type
                # transaction type heuristics
                transaction_type = ai_result.get('transaction_type') or ''
                # shares pick best available numeric
                shares_amount = ai_result.get('shares_acquired') or ai_result.get('shares_after') or ai_result.get('shares_disposed') or 0

                # Build formatted analysis output
                dt_code = (document_type_ai or document_type).upper()
                if dt_code in ('APPENDIX_3Y',):
                    analysis_output = self.format_analysis_output(ai_result)
                elif dt_code in ('APPENDIX_3X',):
                    analysis_output = self.format_appendix_3x_output(ai_result)
                elif dt_code in ('FORM_604',):
                    analysis_output = self.format_form_604_output(ai_result)
                else:
                    # generic
                    analysis_output = self.format_analysis_output(ai_result)

            # Save analysis file if present
            if analysis_output:
                analysis_filename = os.path.join(self.download_dir, f"{asx_code}_AI_ANALYSIS_{int(time.time())}.txt")
                try:
                    with open(analysis_filename, "w", encoding="utf-8") as f:
                        f.write(f"ANALYSIS FOR {asx_code} - {document_title}\n")
                        f.write("="*80 + "\n")
                        f.write(analysis_output + "\n")
                        f.write("="*80 + "\n")
                except Exception:
                    pass

            # Save to DB
            return self.save_announcement(
                announcement_id, asx_code, company_name, document_type, document_title,
                announcement_date, pdf_url, transaction_type, shares_amount, director_name
            )

        except Exception as e:
            print(f"Error processing announcement: {e}")
            return False

    def save_announcement(self, announcement_id: str, asx_code: str, company_name: str,
                          document_type: str, document_title: str, announcement_date: str,
                          pdf_url: Optional[str] = None, transaction_type: Optional[str] = None,
                          shares_amount: Optional[int] = None, director_name: Optional[str] = None) -> bool:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM announcements WHERE id = ?", (announcement_id,))
                if cursor.fetchone():
                    cursor.execute('''
                        UPDATE announcements SET
                        asx_code = ?, company_name = ?, document_type = ?, document_title = ?,
                        announcement_date = ?, pdf_url = ?, transaction_type = ?,
                        shares_amount = ?, director_name = ?, processed = 0
                        WHERE id = ?
                    ''', (asx_code, company_name, document_type, document_title,
                          announcement_date, pdf_url, transaction_type,
                          shares_amount, director_name, announcement_id))
                else:
                    cursor.execute('''
                        INSERT INTO announcements (
                            id, asx_code, company_name, document_type, document_title,
                            announcement_date, pdf_url, transaction_type,
                            shares_amount, director_name, processed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ''', (announcement_id, asx_code, company_name, document_type, document_title,
                          announcement_date, pdf_url, transaction_type, shares_amount, director_name))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving announcement: {e}")
            return False

# ---------- If run as script ----------
def _prompt_data_source() -> str:
    print("Select announcement source:")
    print("  1) JSON feed (Markit Digital)")
    print("  2) HTML scrape (requires OpenAI key for PDF parsing)")
    choice = input("Enter choice [1/2]: ").strip()
    if choice == "2":
        return "html"
    if choice.lower() in ("auto", "a"):
        return "auto"
    return "json"


def main():
    scraper = ASXScraper()
    source = _prompt_data_source()
    if source == "html" and not scraper.openai_api_key:
        print("Warning: HTML mode needs OPENAI_API_KEY configured to parse PDFs.")

    print(f"Fetching announcements using {source.upper()} source...")
    anns = scraper.fetch_announcements(limit=20, source=source)
    processed = 0
    for a in anns:
        if scraper.process_announcement(a):
            processed += 1
    print(f"Processed {processed}/{len(anns)} announcements.")

if __name__ == "__main__":
    main()
