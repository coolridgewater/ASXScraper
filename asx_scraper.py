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
from pdf2image import convert_from_path, convert_from_bytes
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
        # Option: avoid persisting PDFs locally; default false
        def _parse_bool(val: Optional[str]) -> bool:
            if val is None:
                return False
            v = val.strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        self.keep_pdfs = _parse_bool(os.getenv("ASX_KEEP_PDFS", "false"))

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

            # New fields for structured extraction results (trim legacy columns)
            try:
                cursor.execute("SELECT consideration FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN consideration TEXT")
            try:
                cursor.execute("SELECT extracted_json FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN extracted_json TEXT")
            try:
                cursor.execute("SELECT consideration_type FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN consideration_type TEXT")
            # Explicit buy/sell columns
            try:
                cursor.execute("SELECT shares_acquired FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN shares_acquired INTEGER")
            try:
                cursor.execute("SELECT shares_disposed FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN shares_disposed INTEGER")

            # Ensure submission_date exists for storing form notice date
            try:
                cursor.execute("SELECT submission_date FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN submission_date TEXT")
            # Ensure date_of_change exists for storing form's date of change
            try:
                cursor.execute("SELECT date_of_change FROM announcements LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE announcements ADD COLUMN date_of_change TEXT")
            # Ensure numeric consideration and net change
            # Deprecated fields (no longer used): consideration_cents, net_change

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
                # Apply allowlist filter if only_targets
                if self.only_targets:
                    flt = [a for a in anns if self.is_target_announcement(a.get('headline',''))]
                    return flt[:limit]
                return anns[:limit]
            if source == "json":
                print("[info] JSON feed returned no announcements; falling back to HTML.")

        html_anns = self._fetch_announcements_html(limit=limit)
        print(f"[debug] HTML scrape returned {len(html_anns)} announcements (limit {limit}).")
        # HTML fetch method already applies only_targets when building rows.
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
        """HTML scraper. Tries the current ASX announcements page, then falls back to legacy table."""
        announcements: List[Dict] = []
        # 1) Try the user-specified announcements page
        try:
            modern_url = "https://www.asx.com.au/markets/trade-our-cash-market/announcements"
            resp = self.session.get(modern_url, timeout=20)
            print(f"[debug] HTML fetch {modern_url} status={resp.status_code}")
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a'):
                    if len(announcements) >= limit:
                        break
                    href = a.get('href') or ''
                    headline = a.get_text(strip=True) or ''
                    if not href or not headline:
                        continue
                    # Resolve to PDF if possible
                    pdf_url = None
                    hlow = href.lower()
                    if hlow.endswith('.pdf') or 'announcements.asx.com.au' in hlow:
                        pdf_url = urljoin(self.base_url, href)
                    else:
                        pdf_url = self._resolve_pdf_url(href)
                    if not pdf_url:
                        continue
                    # Try to infer code/date from nearby text (best effort)
                    code = ''
                    date_str = ''
                    # Look at parent text for a code (3-5 uppercase letters) and a date dd/mm/yyyy
                    parent_text = a.parent.get_text(" ", strip=True) if a.parent else ''
                    mcode = re.search(r"\b([A-Z]{2,5})\b", parent_text)
                    mdate = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", parent_text)
                    if mcode:
                        code = mcode.group(1)
                    if mdate:
                        date_str = mdate.group(1)
                    # Only accept if we have minimal fields
                    if not code or not date_str:
                        continue
                    doc_id = self._extract_ids_id(href) or self._extract_ids_id(pdf_url) or self.generate_unique_id(code, headline, date_str)
                    ann = {'code': code, 'headline': headline, 'date': date_str, 'id': doc_id, 'url': pdf_url}
                    if self.only_targets:
                        if self.is_target_announcement(headline):
                            announcements.append(ann)
                            if self.debug_fetch:
                                print(f"[ASX HTML modern] match: {code} | {headline}")
                    else:
                        announcements.append(ann)
                if announcements:
                    return announcements
        except Exception:
            pass

        # 2) Fallback to legacy table endpoint
        try:
            url = "https://www.asx.com.au/asx/v2/statistics/todayAnns.do"
            resp = self.session.get(url, timeout=20)
            print(f"[debug] HTML fetch {url} status={resp.status_code}")
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find("table")
            if not table:
                print("[debug] No <table> found in legacy HTML response.")
                return []
            rows = table.find_all("tr")
            print(f"[debug] Parsed {len(rows)} table rows from legacy HTML.")
            for row in rows:
                if len(announcements) >= limit:
                    break
                headers = row.find_all("th")
                if headers:
                    continue
                try:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        code = cells[0].get_text(strip=True)
                        date_cell = cells[1].get_text(" ", strip=True)
                        date_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", date_cell)
                        date_str = date_match.group(1) if date_match else date_cell
                        anchor_cell = cells[-1] if cells else None
                        headline_cell = anchor_cell.find('a') if anchor_cell else None
                        headline = ''
                        if headline_cell:
                            headline_parts = list(headline_cell.stripped_strings)
                            headline = headline_parts[0] if headline_parts else ''
                        doc_link = headline_cell.get('href') if headline_cell else ''
                        pdf_url = self._resolve_pdf_url(doc_link)
                        if self.debug_fetch:
                            print(f"[debug] Legacy row code={code} date={date_str} headline={headline} link={doc_link} pdf={pdf_url}")
                        if not pdf_url:
                            continue
                        doc_id = self._extract_ids_id(doc_link) or self._extract_ids_id(pdf_url) or self.generate_unique_id(code, headline, date_str or str(time.time()))
                        ann = {'code': code, 'headline': headline, 'date': date_str, 'id': doc_id, 'url': pdf_url}
                        if self.only_targets:
                            if self.is_target_announcement(headline):
                                announcements.append(ann)
                        else:
                            announcements.append(ann)
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

        # Regex patterns capturing common variations (exclude 3X/3Z and initial/final director)
        patterns = [
            r"appendix\s*3y",
            r"change\s+of\s+director",  # covers 3Y phrasing
            r"director[^\n]*interest",   # covers 3Y phrasing
            r"form\s*603",
            r"form\s*604",
            r"form\s*605",
            r"substantial\s+(holder|shareholder|holding)",
        ]
        # Keyword fallback including user-provided extras, with explicit phrases
        keywords = [
            'appendix 3y',
            "change of director", "director's interest",
            'form 603', 'form 604', 'form 605',
            'substantial holder', 'substantial shareholder', 'substantial holding',
            # Explicit notice phrases (normalized quotes already handled above)
            "change of director's interest notice",
            "notice of change of interests of substantial holder",
            "substantial shareholder notice"
        ] + self.extra_keywords

        if any(re.search(p, h, re.I) for p in patterns):
            return True
        return any(k in h for k in keywords)

    def determine_document_type(self, headline: str) -> str:
        """
        Heuristically determine document type from headline text.
        """
        h = (headline or "").lower()
        if "appendix 3y" in h or "change of director" in h or "director's interest" in h:
            return "APPENDIX_3Y"
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
    def fetch_pdf(self, url: str):
        """Return (pdf_bytes, saved_path or None). Saves only if keep_pdfs is True."""
        if not url:
            return b"", None
        filepath = None
        try:
            resp = self.session.get(url, timeout=30, allow_redirects=True, stream=True)
            if resp.status_code != 200:
                return b"", None
            content_type = (resp.headers.get("content-type") or "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return b"", None
            content = resp.content or b""
            if self.keep_pdfs and content:
                timestamp = int(time.time())
                filename = f"asx_document_{timestamp}.pdf"
                filepath = os.path.join(self.download_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(content)
            return content, filepath
        except Exception:
            try:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
            return b"", None

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

    def extract_text_from_pdf(self, pdf_path: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> str:
        # Primary extraction with pdfplumber, fallback to PyPDF2 minimal pages
        try:
            text_parts = []
            if pdf_bytes is not None:
                import io
                pdf_obj = pdfplumber.open(io.BytesIO(pdf_bytes))
            else:
                pdf_obj = pdfplumber.open(pdf_path)
            with pdf_obj as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts).strip()
        except Exception:
            try:
                text = ""
                if pdf_bytes is not None:
                    import io
                    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                else:
                    with open(pdf_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                for i in range(min(5, len(reader.pages))):
                    ptext = reader.pages[i].extract_text()
                    if ptext:
                        text += ptext + "\n"
                return text.strip()
            except Exception:
                return ""

    def _ocr_pdf_to_text(self, pdf_path: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> str:
        # Convert PDF pages to images and run pytesseract
        try:
            if pdf_bytes is not None:
                images = convert_from_bytes(pdf_bytes)
            else:
                images = convert_from_path(pdf_path)
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_text).strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    # ---------- Rule-based parsing (Appendix 3Y and similar) ----------
    def _find_after_label(self, lines, label_patterns):
        """Find a value on the same line after a label, or on the next non-empty line."""
        for i, line in enumerate(lines):
            l = line.strip()
            for pat in label_patterns:
                m = re.search(pat, l, flags=re.IGNORECASE)
                if m:
                    after = l[m.end():].strip(" :-\t")
                    if after:
                        return after
                    for j in range(i+1, min(i+4, len(lines))):
                        nxt = lines[j].strip()
                        if nxt:
                            return nxt
        return ""

    def _clean_person_name(self, s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        name = s.strip()
        if not name:
            return None
        # normalize curly quotes/dashes
        name = name.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")
        # remove common noise
        noise_patterns = [
            r"(?i)interest\s+notice",
            r"(?i)change\s+of\s+director.*",
            r"(?i)appendix\s*3y.*",
            r"(?i)notice",
        ]
        for pat in noise_patterns:
            name = re.sub(pat, "", name).strip(" -:\t")
        # Keep only plausible characters
        name = re.sub(r"[^A-Za-z'\-\.\s]", " ", name)
        name = re.sub(r"\s{2,}", " ", name).strip()
        # Heuristic: ignore if contains no letter or is too short
        if not re.search(r"[A-Za-z]", name) or len(name) < 2:
            return None
        return name

    def _to_int(self, s: str) -> Optional[int]:
        if not s:
            return None
        try:
            ns = re.sub(r"[^0-9\-]", "", s)
            if not ns:
                return None
            return int(ns)
        except Exception:
            return None

    def _money_to_cents(self, s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        val = (s or '').strip().lower()
        if val in ('', 'nil', 'n/a', 'na', '-'):
            return None
        clean = re.sub(r"[^0-9\.]", "", s)
        if not clean:
            return None
        try:
            if clean.count('.') > 1:
                first = clean.find('.')
                clean = clean[:first+1] + clean[first+1:].replace('.', '')
            return int(round(float(clean) * 100))
        except Exception:
            return None

    def _to_money(self, s: str) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"([$€£]?\s*[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", s)
        if m:
            return m.group(1).replace(" ", "")
        return None

    def parse_text_fields(self, text: str, doc_title: Optional[str] = None) -> Dict:
        """Extract structured fields from Appendix 3Y-like text using regex heuristics."""
        res: Dict[str, Optional[object]] = {
            "company_name": None,
            "director_name": None,
            "shares_acquired": None,
            "shares_disposed": None,
            "shares_after": None,
            "consideration": None,
            "consideration_type": None,  # 'total_value' or 'unit_price'
            "price_per_share": None,
            "submission_date": None,
            "date_of_change": None,
        }

        if not text:
            return res
        t = text.replace("\r", "\n").replace("\xa0", " ")
        t = re.sub(r"\n{3,}", "\n\n", t)
        lines = [ln.rstrip() for ln in t.split("\n") if ln is not None]

        company = self._find_after_label(lines, [r"^\s*name\s+of\s+entity\s*:?\s*", r"^\s*entity\s*:?\s*"])
        if not company:
            for ln in lines[:10]:
                if re.search(r"\b(ltd|limited)\b", ln, flags=re.IGNORECASE):
                    company = ln.strip(" :-\t")
                    break
            if not company and doc_title:
                m = re.search(r"^(.*?)[-–:]*\s*(appendix|change of director|director)", doc_title, flags=re.IGNORECASE)
                if m:
                    company = m.group(1).strip()
        res["company_name"] = company or None

        # Prefer a direct single-line regex across the whole text for robustness
        director = None
        mdir = re.search(r"(?im)^\s*name\s+of\s+director\(s\)?\s*:?\s*(.+)$", t)
        if not mdir:
            mdir = re.search(r"(?im)^\s*name\s+of\s+the\s+director\(s\)?\s*:?\s*(.+)$", t)
        if mdir:
            director = mdir.group(1).strip()
        else:
            director = self._find_after_label(lines, [
                r"^\s*name\s+of\s+director\(s\)?\s*:?\s*",
                r"^\s*name\s+of\s+the\s+director\(s\)?\s*:?\s*",
            ])
            if not director:
                # Line-based fallback: strip label prefix from the same line
                for line in lines:
                    if re.search(r"(?i)name\s+of\s+director", line):
                        cand = re.sub(r"(?i)^.*name\s+of\s+director\(s\)?\s*:?\s*", "", line).strip()
                        if cand:
                            director = cand
                            break
        if director:
            # Strip any lingering label prefix if present
            # Be robust to different spacing/case
            parts = re.split(r"(?i)name\s+of\s+director\(s\)?\s*:?\s*", director, maxsplit=1)
            if len(parts) > 1:
                director = parts[1].strip()
            else:
                low = director.lower()
                key = "name of director"
                p = low.find(key)
                if p != -1:
                    director = director[p+len(key):].strip()
            # remove titles and clean
            director = re.sub(r"\b(mr\.|mrs\.|ms\.|dr\.|hon\.)\b\s*", "", director, flags=re.IGNORECASE).strip()
            director = self._clean_person_name(director) or director
        # Fallback from document title patterns
        if (not director or not self._clean_person_name(director)) and doc_title:
            # Common headline: "Appendix 3Y - First Last" or "Change of Director's Interest Notice - First Last"
            m = re.search(r"(?i)appendix\s*3y\s*[-–:\u2013\u2014]\s*(.+)$", doc_title)
            if not m:
                m = re.search(r"(?i)change\s+of\s+director.*?[-–:\u2013\u2014]\s*(.+)$", doc_title)
            if m:
                cand = self._clean_person_name(m.group(1))
                if cand:
                    director = cand
        res["director_name"] = director or None

        num_acq = self._find_after_label(lines, [r"number\s+acquired\s*:?\s*", r"no\.?\s+acquired\s*:?\s*", r"securities\s+acquired\s*:?\s*"])
        num_dis = self._find_after_label(lines, [r"number\s+disposed\s*:?\s*", r"no\.?\s+disposed\s*:?\s*", r"securities\s+disposed\s*:?\s*"])
        ia = self._to_int(num_acq)
        idp = self._to_int(num_dis)
        res["shares_acquired"] = ia
        res["shares_disposed"] = idp

        cons = self._find_after_label(lines, [r"value\s*/?\s*consideration\s*\(?.*?\)?\s*:?\s*", r"consideration\s*:?\s*"])
        cons_money = self._to_money(cons)
        if cons_money:
            res["consideration"] = cons_money
        elif cons:
            res["consideration"] = cons

        # Price per share patterns
        pps = self._find_after_label(lines, [
            r"price\s+per\s+(security|share)\s*:?\s*",
            r"price\s*/\s*consideration\s*per\s+(security|share)\s*:?\s*",
            r"unit\s+price\s*:?\s*",
            r"price\s+paid\s+per\s+(security|share)\s*:?\s*",
        ])
        pps_money = self._to_money(pps)
        if pps_money:
            res["price_per_share"] = pps_money
            res["consideration_type"] = "per_share"
        elif cons_money:
            res["consideration_type"] = "total_value"

        after = self._find_after_label(lines, [r"no\.?\s+of\s+securities\s+held\s+after\s+change\s*:?\s*", r"number\s+held\s+after\s+change\s*:?\s*"])
        res["shares_after"] = self._to_int(after)

        # Date of form submission (often "Date of this notice")
        date_label_val = self._find_after_label(
            lines,
            [
                r"date\s+of\s+this\s+notice\s*:?\s*",
                r"date\s+of\s+notice\s*:?\s*",
                r"date\s+of\s+submission\s*:?\s*",
                r"lodgement\s+date\s*:?\s*",
            ],
        )
        def _parse_date(s: str) -> Optional[str]:
            if not s:
                return None
            s = s.strip()
            for fmt in ("%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    from datetime import datetime
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except Exception:
                    continue
            # Try to find a date substring inside s
            m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", s)
            if m:
                try:
                    from datetime import datetime
                    return datetime.strptime(m.group(1), "%d/%m/%Y").strftime("%Y-%m-%d")
                except Exception:
                    pass
            return None
        res["submission_date"] = _parse_date(date_label_val)

        # Date of change (Appendix 3Y label)
        change_date_label_val = self._find_after_label(
            lines,
            [
                r"date\s+of\s+change\s*:?\s*",
                r"date\s+of\s+changes\s*:?\s*",
            ],
        )
        res["date_of_change"] = _parse_date(change_date_label_val)

        return res

    def parse_pdf_structured(self, pdf_path: Optional[str] = None, doc_title: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> Dict:
        """Read a PDF, try text extraction (with OCR fallback), and extract structured fields."""
        text = self.extract_text_from_pdf(pdf_path=pdf_path, pdf_bytes=pdf_bytes)
        if len((text or "").strip()) < 120:
            ocr_text = self._ocr_pdf_to_text(pdf_path=pdf_path, pdf_bytes=pdf_bytes)
            text = (ocr_text or text or "")
        text = self._normalize_text(text)
        if not text:
            return {}
        data = self.parse_text_fields(text, doc_title=doc_title)
        data["raw_text_excerpt"] = text[:2000]
        return data

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

    def parse_pdf_with_ai(self, pdf_path: Optional[str] = None, doc_title: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> Dict:
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
        text = self.extract_text_from_pdf(pdf_path=pdf_path, pdf_bytes=pdf_bytes)
        if len((text or "").strip()) < 200:
            # fallback to OCR
            ocr_text = self._ocr_pdf_to_text(pdf_path=pdf_path, pdf_bytes=pdf_bytes)
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
            # Normalize document_type to known codes (exclude 3X/3Z)
            dt = result.get("document_type")
            if isinstance(dt, str):
                up = dt.upper()
                if "3Y" in up:
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

    # (Removed Appendix 3X formatter by request)

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

            # If there's a PDF, fetch it (in-memory by default)
            pdf_path = None
            pdf_bytes = None
            if pdf_url:
                pdf_bytes, saved_path = self.fetch_pdf(pdf_url)
                if self.keep_pdfs:
                    pdf_path = saved_path
            else:
                # no PDF — still attempt to call AI with the headline + metadata (very limited)
                # create a small temporary text file for AI to parse
                pdf_path = None

            # Always use AI parser as requested
            ai_result = None
            if pdf_path or pdf_bytes:
                ai_result = self.parse_pdf_with_ai(pdf_path=pdf_path, pdf_bytes=pdf_bytes, doc_title=document_title)
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

            # Rule-based structured extraction (offline)
            structured_data = {}
            if pdf_path or pdf_bytes:
                try:
                    structured_data = self.parse_pdf_structured(pdf_path=pdf_path, pdf_bytes=pdf_bytes, doc_title=document_title) or {}
                except Exception:
                    structured_data = {}

            # Merge heuristic + AI results and map to fields
            merged = {}
            if structured_data:
                merged.update({k: v for k, v in structured_data.items() if v not in (None, "")})
            if ai_result:
                merged.update({k: v for k, v in ai_result.items() if v not in (None, "")})

            consideration = None
            shares_acquired_val = None
            shares_disposed_val = None
            consideration_cents_val = None
            submission_date_val = None
            date_of_change_val = None
            consideration_type_val = None
            if merged:
                company_name = merged.get('company_name') or company_name
                director_name = merged.get('director_name') or director_name
                document_type_ai = merged.get('document_type') or document_type
                # transaction type heuristics
                transaction_type = merged.get('transaction_type') or ''
                # shares pick best available numeric (no dependency on shares_after)
                shares_amount = merged.get('shares_acquired') or merged.get('shares_disposed') or 0
                consideration = merged.get('consideration') if isinstance(merged.get('consideration'), str) else None
                consideration_type_val = merged.get('consideration_type') if isinstance(merged.get('consideration_type'), str) else None
                # explicit acquired/disposed
                try:
                    if merged.get('shares_acquired') is not None:
                        shares_acquired_val = int(merged.get('shares_acquired'))
                except Exception:
                    shares_acquired_val = None
                try:
                    if merged.get('shares_disposed') is not None:
                        shares_disposed_val = int(merged.get('shares_disposed'))
                except Exception:
                    shares_disposed_val = None
                # derived metrics
                price_per_share = merged.get('price_per_share') if isinstance(merged.get('price_per_share'), str) else None
                # Normalize consideration type alias
                if consideration_type_val and consideration_type_val.lower() == 'unit_price':
                    consideration_type_val = 'per_share'
                if consideration_type_val == 'per_share' and price_per_share and (shares_acquired_val is not None):
                    pps_cents = self._money_to_cents(price_per_share)
                    if pps_cents is not None:
                        consideration_cents_val = pps_cents * int(shares_acquired_val or 0)
                if consideration_cents_val is None and consideration:
                    consideration_cents_val = self._money_to_cents(consideration)
                # submission date: prefer parsed from PDF, else announcement_date
                sd = merged.get('submission_date')
                if isinstance(sd, str) and sd:
                    # ensure yyyy-mm-dd if possible
                    try:
                        submission_date_val = datetime.strptime(sd, '%Y-%m-%d').strftime('%Y-%m-%d')
                    except Exception:
                        try:
                            submission_date_val = datetime.strptime(sd, '%d/%m/%Y').strftime('%Y-%m-%d')
                        except Exception:
                            submission_date_val = sd
                if not submission_date_val:
                    submission_date_val = announcement_date
                # date of change: prefer parsed from PDF
                dc = merged.get('date_of_change')
                if isinstance(dc, str) and dc:
                    try:
                        date_of_change_val = datetime.strptime(dc, '%Y-%m-%d').strftime('%Y-%m-%d')
                    except Exception:
                        try:
                            date_of_change_val = datetime.strptime(dc, '%d/%m/%Y').strftime('%Y-%m-%d')
                        except Exception:
                            date_of_change_val = dc

                # Build formatted analysis output
                dt_code = (document_type_ai or document_type).upper()
                if dt_code in ('APPENDIX_3Y',):
                    analysis_output = self.format_analysis_output(merged)
                elif dt_code in ('FORM_604',):
                    analysis_output = self.format_form_604_output(merged)
                else:
                    # generic
                    analysis_output = self.format_analysis_output(merged)

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

            # Save to DB (base fields), then persist structured fields if present
            ok = self.save_announcement(
                announcement_id, asx_code, company_name, document_type, document_title,
                announcement_date, pdf_url, transaction_type, shares_amount, director_name,
                submission_date=submission_date_val, date_of_change=date_of_change_val
            )
            if ok:
                try:
                    with self.get_db_connection() as conn:
                        cur = conn.cursor()
                        extracted_json = None
                        try:
                            extracted_json = json.dumps(merged) if merged else None
                        except Exception:
                            extracted_json = None
                        cur.execute("""
                            UPDATE announcements
                            SET consideration = COALESCE(?, consideration),
                                consideration_type = COALESCE(?, consideration_type),
                                shares_acquired = COALESCE(?, shares_acquired),
                                shares_disposed = COALESCE(?, shares_disposed),
                                consideration_cents = COALESCE(?, consideration_cents),
                                extracted_json = COALESCE(?, extracted_json)
                            WHERE id = ?
                        """, (consideration, consideration_type_val, shares_acquired_val, shares_disposed_val, consideration_cents_val, extracted_json, announcement_id))
                        conn.commit()
                except Exception:
                    pass
            return ok

        except Exception as e:
            print(f"Error processing announcement: {e}")
            return False

    def save_announcement(self, announcement_id: str, asx_code: str, company_name: str,
                          document_type: str, document_title: str, announcement_date: str,
                          pdf_url: Optional[str] = None, transaction_type: Optional[str] = None,
                          shares_amount: Optional[int] = None, director_name: Optional[str] = None,
                          submission_date: Optional[str] = None, date_of_change: Optional[str] = None) -> bool:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM announcements WHERE id = ?", (announcement_id,))
                if cursor.fetchone():
                    cursor.execute('''
                        UPDATE announcements SET
                        asx_code = ?, company_name = ?, document_type = ?, document_title = ?,
                        announcement_date = ?, submission_date = ?, date_of_change = ?, pdf_url = ?, transaction_type = ?,
                        shares_amount = ?, director_name = ?, processed = 0
                        WHERE id = ?
                    ''', (asx_code, company_name, document_type, document_title,
                          announcement_date, submission_date, date_of_change, pdf_url, transaction_type,
                          shares_amount, director_name, announcement_id))
                else:
                    cursor.execute('''
                        INSERT INTO announcements (
                            id, asx_code, company_name, document_type, document_title,
                            announcement_date, submission_date, date_of_change, pdf_url, transaction_type,
                            shares_amount, director_name, processed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ''', (announcement_id, asx_code, company_name, document_type, document_title,
                          announcement_date, submission_date, date_of_change, pdf_url, transaction_type, shares_amount, director_name))
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

