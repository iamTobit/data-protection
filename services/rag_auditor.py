"""
RAG Compliance Auditor Service for Kenyan Data Protection Act (ODPC).
Features:
1. Automated Privacy Policy scraping from direct URL or root website.
2. Vector similarity retrieval from Supabase/PostgreSQL pgvector (BAAI/bge-small-en-v1.5).
3. LLM Compliance evaluation supporting OpenAI-compatible endpoints & Hugging Face,
   with robust fallback heuristics for zero-crash developer testing.
"""

import os
import re
import json
import logging
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from database.connection import get_db_connection

load_dotenv()
logger = logging.getLogger(__name__)

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()

# Lazy-loaded embedding model cache
_EMBEDDING_MODEL = None


def get_embedding_model():
    """
    Lazy-loads the BAAI/bge-small-en-v1.5 sentence-transformers model.
    384-dimensional embeddings optimized for legal retrieval.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers embedding model: BAAI/bge-small-en-v1.5...")
            _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.warning(
                "sentence-transformers not yet loaded or torch initializing: %s. "
                "Deterministic 384-dim semantic feature vector fallback will be active.", e
            )
            return None
    return _EMBEDDING_MODEL


def _deterministic_384d_embedding(text: str) -> List[float]:
    """
    Computes a deterministic, normalized 384-dimensional vector from input text.
    Provides a seamless fallback for 384-dim pgvector operations when
    neural weights are downloading or in resource-constrained environments.
    """
    import math
    import hashlib

    dim = 384
    vec = [0.0] * dim
    
    # Tokenize and hash into 384 buckets with sign hashing
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        tokens = [text.lower().strip() or "empty"]
        
    for i, token in enumerate(tokens):
        # Position-aware hashing
        h = int(hashlib.sha256(f"{token}_{i%16}".encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 9) & 1) else -1.0
        vec[idx] += sign

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec


def generate_embedding(text: str) -> List[float]:
    """
    Generates a 384-dimensional normalized embedding for a given text.
    Uses BAAI/bge-small-en-v1.5 when sentence-transformers is loaded,
    falling back seamlessly to deterministic 384-dim projection if needed.
    
    Args:
        text: Query or document string.
        
    Returns:
        List[float]: 384-dim vector.
    """
    clean_text = text.replace("\n", " ").strip()
    try:
        model = get_embedding_model()
        if model is not None:
            embedding = model.encode(clean_text, normalize_embeddings=True)
            return embedding.tolist()
    except Exception as e:
        logger.warning("Neural embedding calculation error (%s), using 384d fallback.", e)
        
    return _deterministic_384d_embedding(clean_text)


# -------------------------------------------------------------------------
# 1. Web Scraping & Privacy Policy Auto-Discovery
# -------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

PRIVACY_KEYWORDS = [
    "privacy", "privacy policy", "data protection", "data privacy",
    "privacy notice", "terms and privacy", "legal/privacy", "privacy-policy"
]


def clean_html_text(html_content: str) -> str:
    """Extracts visible, cleaned plain text from HTML, discarding boilerplate."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove irrelevant tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "svg", "noscript", "form", "iframe"]):
        tag.decompose()
        
    # Extract text from paragraphs, divs, and section headers
    text_blocks = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = element.get_text(separator=" ", strip=True)
        if txt and len(txt) > 20:
            text_blocks.append(txt)
            
    if not text_blocks:
        # Fallback to broad text extraction
        return soup.get_text(separator="\n", strip=True)
        
    return "\n\n".join(text_blocks)


def scrape_policy_from_url(target_url: str, timeout: int = 15) -> Dict[str, str]:
    """
    Fetches a business website. If the provided URL is a root or landing page,
    automatically detects and follows the Privacy Policy link.
    
    Args:
        target_url: URL to scrape (e.g., https://example.co.ke or https://example.co.ke/privacy).
        timeout: HTTP request timeout in seconds.
        
    Returns:
        Dict with keys:
            - "source_url": Final scraped URL
            - "title": Page title
            - "text": Clean extracted policy text
            
    Raises:
        ValueError: If URL is invalid, unreachable, or returns insufficient text.
    """
    if not target_url.startswith(("http://", "https://")):
        target_url = "https://" + target_url

    parsed = urlparse(target_url)
    if not parsed.netloc:
        raise ValueError(f"Invalid URL provided: '{target_url}'")

    logger.info("Scraping target URL: %s", target_url)

    try:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        
        response = session.get(target_url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        page_title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
        
        # Check if the current page already looks like a dedicated privacy policy
        is_direct_policy = any(k in response.url.lower() or k in page_title.lower() for k in ["privacy", "data-protection"])
        clean_text = clean_html_text(response.text)
        
        # If it's a home page or direct text is small, look for a Privacy Policy link
        if not is_direct_policy or len(clean_text) < 400:
            privacy_link = None
            
            # 1. Search <a> tags by text or href
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                link_text = a_tag.get_text(strip=True).lower()
                
                # Check for privacy keywords in text or href
                if any(kw in link_text for kw in PRIVACY_KEYWORDS) or any(kw in href.lower() for kw in ["privacy", "data-protection"]):
                    privacy_link = urljoin(response.url, href)
                    break
                    
            if privacy_link and privacy_link != response.url:
                logger.info("Discovered Privacy Policy link at: %s. Fetching...", privacy_link)
                try:
                    policy_resp = session.get(privacy_link, timeout=timeout, allow_redirects=True)
                    policy_resp.raise_for_status()
                    policy_soup = BeautifulSoup(policy_resp.text, "html.parser")
                    policy_text = clean_html_text(policy_resp.text)
                    
                    if len(policy_text) >= 300:
                        title = policy_soup.title.string.strip() if policy_soup.title and policy_soup.title.string else page_title
                        return {
                            "source_url": policy_resp.url,
                            "title": title,
                            "text": policy_text
                        }
                except Exception as link_err:
                    logger.warning("Could not fetch discovered privacy link %s: %s", privacy_link, link_err)

        if len(clean_text) < 150:
            raise ValueError(
                f"Scraped page contains insufficient text ({len(clean_text)} characters) "
                f"to perform a meaningful legal audit. Please paste the Privacy Policy text directly."
            )

        return {
            "source_url": response.url,
            "title": page_title,
            "text": clean_text
        }

    except requests.exceptions.Timeout:
        raise ValueError(f"Connection timed out while trying to reach {target_url}. Please check the URL or paste text directly.")
    except requests.exceptions.HTTPError as he:
        status_code = he.response.status_code if he.response is not None else 0
        if status_code in [403, 401]:
            raise ValueError(
                f"🔒 Anti-Bot Protection Detected (HTTP {status_code}): '{parsed.netloc}' uses Cloudflare/Akamai bot protection "
                f"that restricts automated web scrapers. Please copy the policy text from your browser and paste it directly into the 'Paste Policy' tab."
            )
        raise ValueError(f"HTTP error {status_code} accessing {target_url}: {he}. Please paste the policy text directly.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch {target_url}: {e}. Please ensure the site is public or paste the text directly.")


# -------------------------------------------------------------------------
# 2. Vector Search / Legal Knowledge Retrieval
# -------------------------------------------------------------------------

COMPLIANCE_DOMAINS = [
    {
        "domain": "Consent & Lawful Processing",
        "query": "conditions of consent freely given clear withdrawal burden of proof lawful basis for processing personal data Kenya Data Protection Act Section 30 32"
    },
    {
        "domain": "Principles & Data Minimisation",
        "query": "principles of data protection purpose limitation data minimisation accuracy storage limitation lawful fair and transparent Kenya DPA Section 25"
    },
    {
        "domain": "Data Subject Rights",
        "query": "rights of data subject right of access rectification erasure objection automated decision making profiling Kenya Data Protection Act Section 26 40"
    },
    {
        "domain": "Direct Marketing & Opt-out",
        "query": "commercial use of personal data direct marketing consent opt out opt in mechanism Kenya Data Protection Act Section 37"
    },
    {
        "domain": "Data Retention & Storage Limitation",
        "query": "retention of personal data retention period destruction deletion archiving Kenya Data Protection Act Section 39"
    },
    {
        "domain": "Cross-Border Transfers",
        "query": "transfer of personal data outside Kenya appropriate safeguards adequacy decision ODPC approval consent Kenya Data Protection Act Section 48 49"
    },
    {
        "domain": "Breach Notification & Security",
        "query": "notification of personal data breach to Office of the Data Protection Commissioner within 72 hours security measures safeguards Kenya DPA Section 43"
    },
    {
        "domain": "ODPC Registration & Governance",
        "query": "registration of data controllers and processors appointment of Data Protection Officer complaints to Data Commissioner Section 18 24 29"
    }
]


def retrieve_relevant_legal_context(document_text: str, top_k_per_domain: int = 1) -> List[Dict[str, Any]]:
    """
    Queries pgvector across key Kenyan legal compliance domains to retrieve
    the most relevant statutory sections, citations, and guidance notes.
    
    Args:
        document_text: The user's privacy policy text.
        top_k_per_domain: Number of chunks per legal domain.
        
    Returns:
        List of legal chunk records (citation, act_title, section_ref, content).
    """
    retrieved_chunks = []
    seen_ids = set()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for domain_spec in COMPLIANCE_DOMAINS:
                    query_text = f"{domain_spec['query']} {document_text[:200]}"
                    query_embedding = generate_embedding(query_text)

                    # Cosine distance operator in pgvector is <=>
                    # Parameterized query with embedding vector
                    cur.execute(
                        """
                        SELECT id, act_title, section_ref, citation, content, 
                               (embedding <=> %s::vector) AS distance
                        FROM legal_chunks
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s;
                        """,
                        (query_embedding, query_embedding, top_k_per_domain)
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            retrieved_chunks.append({
                                "id": r["id"],
                                "act_title": r["act_title"],
                                "section_ref": r["section_ref"],
                                "citation": r["citation"],
                                "content": r["content"],
                                "distance": float(r["distance"])
                            })
                            
        logger.info("Retrieved %d relevant legal chunks from pgvector.", len(retrieved_chunks))
    except Exception as e:
        logger.warning(
            "Could not query pgvector for legal chunks: %s. "
            "Proceeding with built-in statutory baseline knowledge.", e
        )

    return retrieved_chunks


# -------------------------------------------------------------------------
# 3. LLM Inference & Heuristic Evaluator
# -------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Kenya's leading Data Protection Legal Counsel and Senior Regulatory Auditor specializing in the Office of the Data Protection Commissioner (ODPC) enforcement and the Kenya Data Protection Act, 2019 (DPA 2019) and the Data Protection (General) Regulations, 2021.

Your task is to conduct an authoritative, rigorous compliance audit on the provided organization Privacy Policy or legal contract text.

You MUST evaluate the policy against these mandatory Kenyan legal pillars:
1. Consent & Lawful Basis (Section 30, 32 DPA 2019): Must be freely given, specific, informed, with explicit burden of proof on the controller, and simple withdrawal mechanism.
2. Principles of Data Protection (Section 25 DPA 2019): Transparency, purpose limitation, collection minimization, accuracy, storage limitation.
3. Rights of Data Subjects (Section 26, 40 DPA 2019): Right to be informed, right of access, right of rectification, right to object, right to erasure, and automated decision-making protections.
4. Direct Marketing & Commercial Use (Section 37 DPA 2019): Commercial communication prohibited without prior consent; mandatory opt-out mechanism with no cost.
5. Retention & Storage Limitation (Section 39 DPA 2019): Vague "as long as necessary" phrases violate Kenyan law; explicit retention periods or criteria are mandatory.
6. Cross-Border Data Transfers (Section 48, 49 DPA 2019): Transfers outside Kenya require appropriate safeguards, adequacy, or explicit consent + ODPC notification.
7. Security & Breach Notification (Section 43 DPA 2019): Mandatory notification of personal data breaches to the ODPC within 72 hours.
8. ODPC Contact & DPO (Section 24, 29 DPA 2019): Right to lodge complaints with the Data Commissioner (complaints@odpc.go.ke / info@odpc.go.ke).

Return a strictly valid JSON object with EXACTLY this structure:
{
  "risk_level": "High" | "Medium" | "Low",
  "overall_score": <number between 1 and 100 representing compliance percentage>,
  "executive_summary": "<Detailed, 2-3 paragraph professional legal assessment of the business's ODPC posture>",
  "violations": [
    {
      "id": 1,
      "title": "<Clear, concise title of violation>",
      "severity": "Critical" | "High" | "Medium" | "Low",
      "citation": "<Exact statutory citation, e.g. Kenya Data Protection Act 2019, Section 37(1)>",
      "explanation": "<Specific analysis explaining why the policy text violates Kenyan law and risk of ODPC statutory fines up to KES 5,000,000>",
      "recommended_fix": "<Actionable, legal drafting instructions to remedy the violation>"
    }
  ],
  "compliant_policy_draft": "<A complete, comprehensive, beautifully structured replacement Privacy Policy tailored specifically to this business entity, ready for immediate publishing. Must include full sections for controller details, legal bases, exact retention periods, data subject rights with ODPC complaints channels, cross-border safeguards, cookie disclosures, and DPO designation.>"
}
Do not wrap your response in markdown code blocks like ```json ... ```. Output raw JSON only."""


def _audit_with_llm(document_text: str, legal_chunks: List[Dict[str, Any]], entity_name: str) -> Dict[str, Any]:
    """Calls OpenAI or OpenAI-compatible LLM endpoint."""
    from openai import OpenAI

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL if LLM_BASE_URL else None
    )

    legal_context_str = "\n\n".join([
        f"--- {chunk['citation']} ({chunk['act_title']}) ---\n{chunk['content']}"
        for chunk in legal_chunks
    ]) if legal_chunks else "Standard Kenyan Data Protection Act 2019 provisions."

    user_prompt = f"""Target Entity / Business Name: {entity_name}

RELEVANT KENYAN STATUTORY CHUNKS RETRIEVED FROM ODPC DATABASE:
{legal_context_str}

SUBMITTED PRIVACY POLICY TEXT TO AUDIT:
\"\"\"
{document_text[:8000]}
\"\"\"

Perform the rigorous Kenyan ODPC legal audit now. Output valid JSON only."""

    logger.info("Calling LLM model '%s' for compliance audit...", LLM_MODEL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"} if "gpt-4" in LLM_MODEL or "gpt-3.5" in LLM_MODEL else None
    )

    raw_content = response.choices[0].message.content.strip()
    
    # Strip markdown fences if present
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
        raw_content = re.sub(r"\n?```$", "", raw_content)

    return json.loads(raw_content)


def _audit_with_heuristics(document_text: str, legal_chunks: List[Dict[str, Any]], entity_name: str) -> Dict[str, Any]:
    """
    Robust statutory compliance engine that evaluates Kenyan DPA 2019 criteria
    deterministically using pattern matching and statutory checks.
    Used when no LLM API key is provided, or as an instant resilient fallback.
    """
    text_lower = document_text.lower()
    violations = []
    vid = 1

    # Check 1: ODPC complaints channel & Data Commissioner mentions
    has_odpc = any(term in text_lower for term in ["odpc", "data commissioner", "office of the data protection commissioner", "complaints@odpc.go.ke"])
    if not has_odpc:
        violations.append({
            "id": vid,
            "title": "Absence of ODPC Supervisory Authority Complaint Channel",
            "severity": "Critical",
            "citation": "Kenya Data Protection Act, 2019, Section 26(e) & Section 56",
            "explanation": (
                "The policy fails to inform Kenyan data subjects of their statutory right to lodge formal complaints "
                "with the Office of the Data Protection Commissioner (ODPC) at complaints@odpc.go.ke or info@odpc.go.ke. "
                "Under the DPA 2019, transparency regarding regulatory recourse is a mandatory disclosure."
            ),
            "recommended_fix": (
                "Insert an explicit section titled 'Your Right to Lodge a Complaint with the ODPC' directing data subjects "
                "to the Office of the Data Protection Commissioner, Kenya (P.O. Box 30920-00100, Nairobi, Website: www.odpc.go.ke)."
            )
        })
        vid += 1

    # Check 2: Direct Marketing opt-out (Section 37)
    has_marketing_optout = ("direct marketing" in text_lower or "marketing" in text_lower) and any(
        term in text_lower for term in ["opt out", "opt-out", "unsubscribe", "withdraw consent", "without charge"]
    )
    if not has_marketing_optout:
        violations.append({
            "id": vid,
            "title": "Unlawful Commercial Use & Missing Direct Marketing Opt-Out Mechanism",
            "severity": "Critical",
            "citation": "Kenya Data Protection Act, 2019, Section 37 & Regulations 13-14",
            "explanation": (
                "Section 37 prohibits the use of personal data for commercial direct marketing unless the data controller "
                "has obtained prior explicit consent and provided an unburdened, cost-free opt-out mechanism. Vague clauses "
                "claiming authority to send promotional messages without an affirmative opt-out violate Kenyan law."
            ),
            "recommended_fix": (
                "Draft an explicit opt-out mechanism for all commercial and marketing communications, ensuring users "
                "can withdraw marketing consent at zero cost via an online toggle or automated unsubscribe link."
            )
        })
        vid += 1

    # Check 3: Retention Periods (Section 39)
    vague_retention = any(phrase in text_lower for phrase in ["as long as necessary", "as long as needed", "indefinitely", "reasonable time"])
    has_specific_retention = any(term in text_lower for term in ["retention schedule", "retained for", "years", "months", "days after account closure"])
    if vague_retention or not has_specific_retention:
        violations.append({
            "id": vid,
            "title": "Indefinite & Vague Data Retention Provisions",
            "severity": "High",
            "citation": "Kenya Data Protection Act, 2019, Section 39(1) & Regulation 19",
            "explanation": (
                "Section 39 mandates that personal data must not be kept longer than is necessary for the lawful purpose "
                "for which it was collected. Stating data is kept 'as long as necessary' or failing to state retention schedules "
                "contravenes the storage limitation principle."
            ),
            "recommended_fix": (
                "Specify defined retention periods (e.g. 'Financial and transaction records are retained for 7 years pursuant "
                "to Kenyan tax legislation; session data is purged after 90 days')."
            )
        })
        vid += 1

    # Check 4: Cross-Border Transfers outside Kenya (Section 48, 49)
    mentions_transfers = any(term in text_lower for term in ["overseas", "foreign", "cloud", "aws", "united states", "international transfer", "cross-border"])
    has_safeguards = any(term in text_lower for term in ["appropriate safeguards", "standard contractual clauses", "adequacy decision", "section 48", "consent of the data subject"])
    if mentions_transfers and not has_safeguards:
        violations.append({
            "id": vid,
            "title": "Non-Compliant Cross-Border Data Transfer Disclosures",
            "severity": "High",
            "citation": "Kenya Data Protection Act, 2019, Section 48 & 49",
            "explanation": (
                "Transferring personal data outside Kenya to offshore servers (e.g., US/EU cloud hosting) without demonstrating "
                "proof of appropriate safeguards, an ODPC adequacy determination, or explicit consent violates Part VI of the Act."
            ),
            "recommended_fix": (
                "Document that international data transfers are protected under Standard Contractual Clauses (SCCs) "
                "aligned with ODPC guidance, and obtain explicit consent where adequate legal protections do not exist."
            )
        })
        vid += 1

    # Check 5: 72-Hour Breach Notification (Section 43)
    has_breach_policy = any(term in text_lower for term in ["breach", "security incident", "72 hours", "notify the commissioner", "notifying the odpc"])
    if not has_breach_policy:
        violations.append({
            "id": vid,
            "title": "Omission of Mandatory 72-Hour ODPC Breach Notification Protocols",
            "severity": "Medium",
            "citation": "Kenya Data Protection Act, 2019, Section 43(1)",
            "explanation": (
                "Section 43 imposes a strict legal duty on data controllers to notify the Data Commissioner within 72 hours "
                "of becoming aware of a breach involving personal data that presents a real risk of harm."
            ),
            "recommended_fix": (
                "Incorporate a formal breach response disclosure committing to notify the ODPC within 72 hours and impacted "
                "data subjects without undue delay in the event of high-risk security compromises."
            )
        })
        vid += 1

    # Check 6: Full Data Subject Rights Enumeration (Section 26)
    missing_rights = []
    if "rectification" not in text_lower and "correct" not in text_lower:
        missing_rights.append("Right to Rectification (Section 26(b))")
    if "erasure" not in text_lower and "deletion" not in text_lower and "delete" not in text_lower:
        missing_rights.append("Right to Erasure / Right to be Forgotten (Section 26(c))")
    if "object" not in text_lower and "objection" not in text_lower:
        missing_rights.append("Right to Object (Section 26(d))")
    if "portability" not in text_lower and "data portability" not in text_lower:
        missing_rights.append("Right to Data Portability (Section 26(f))")
    if "automated" not in text_lower and "profiling" not in text_lower:
        missing_rights.append("Right not to be Subjected to Automated Decision-Making (Section 40)")

    if len(missing_rights) >= 2:
        violations.append({
            "id": vid,
            "title": f"Incomplete Enumeration of Data Subject Rights ({len(missing_rights)} rights missing)",
            "severity": "High",
            "citation": "Kenya Data Protection Act, 2019, Section 26 & Section 40",
            "explanation": (
                f"The policy omits several fundamental rights guaranteed to individuals under Section 26 of the Act: "
                f"{', '.join(missing_rights)}. Under Kenyan law, data controllers must provide simple mechanisms to exercise these rights."
            ),
            "recommended_fix": (
                "Provide a comprehensive 'Your Statutory Rights' section clearly detailing all 6 rights under Section 26, "
                "specifying contact details for the appointed Data Protection Officer (DPO) and turnaround response times."
            )
        })
        vid += 1

    # Check 7: Children's Data Protection (Section 33)
    if "child" not in text_lower and "minor" not in text_lower and "consent of the parent" not in text_lower:
        violations.append({
            "id": vid,
            "title": "Absence of Protections for Personal Data of Children",
            "severity": "Medium",
            "citation": "Kenya Data Protection Act, 2019, Section 33",
            "explanation": (
                "Section 33 strictly prohibits processing personal data relating to a child unless prior consent is obtained "
                "from the child's parent or legal guardian, and age verification safeguards are maintained."
            ),
            "recommended_fix": (
                "State the minimum age requirement for using the service (18 years in Kenya) and confirm that children's data "
                "is not knowingly processed without verified parental or legal guardian consent."
            )
        })
        vid += 1

    # Score calculation
    crit_count = sum(1 for v in violations if v["severity"] == "Critical")
    high_count = sum(1 for v in violations if v["severity"] == "High")
    med_count = sum(1 for v in violations if v["severity"] == "Medium")

    deductions = (crit_count * 25) + (high_count * 15) + (med_count * 8)
    overall_score = max(15, min(95, 100 - deductions))

    if crit_count > 0 or overall_score < 50:
        risk_level = "High"
    elif high_count > 0 or overall_score < 75:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Compliant draft generation
    name = entity_name if entity_name and entity_name != "Unknown" else "The Organization"
    compliant_policy = f"""# ODPC-COMPLIANT PRIVACY POLICY
**Governed under the Kenya Data Protection Act, 2019 and Data Protection (General) Regulations, 2021**
*Effective Date: November 2026 | Version 2.0 (ODPC Aligned)*

---

### 1. Introduction and Data Controller Details
Welcome to **{name}** ("we", "us", or "our"). We are registered and operate as a Data Controller / Data Processor under the **Data Protection Act, No. 24 of 2019 (the "Act")** overseen by the Office of the Data Protection Commissioner (ODPC) of the Republic of Kenya.

We respect the privacy of our customers, website visitors, and partners ("Data Subjects") and are committed to processing your personal data lawfully, fairly, and transparently in accordance with **Section 25 of the Act**.

- **Registered Entity**: {name}
- **Physical Address**: Nairobi, Kenya
- **Data Protection Officer (DPO) Contact**: dpo@{name.lower().replace(' ', '')}.co.ke
- **General Inquiries**: privacy@{name.lower().replace(' ', '')}.co.ke

---

### 2. Principles of Data Protection (Section 25)
All personal data collected by {name} is processed strictly pursuant to the eight statutory principles under Kenyan law:
1. Processed lawfully, fairly, and in a transparent manner.
2. Collected for explicit, specified, and legitimate purposes (Purpose Limitation).
3. Adequate, relevant, and limited to what is necessary in relation to the purposes (Data Minimisation).
4. Accurate and kept up to date; reasonable steps are taken to rectify inaccuracies without delay.
5. Kept in a form which permits identification for no longer than is necessary (Storage Limitation).
6. Processed in accordance with the rights of data subjects.
7. Protected against unauthorized processing, loss, or destruction using appropriate technical measures.
8. Transferred outside Kenya only upon verification of adequate legal safeguards.

---

### 3. Categories of Personal Data Collected
We collect the following categories of data directly from you pursuant to **Section 28**:
- **Identity & Contact Data**: Full legal name, national ID / passport number, email address, physical location, and Safaricom / Airtel mobile numbers.
- **Financial & Transaction Data**: M-Pesa transaction identifiers, billing addresses, and payment records.
- **Technical & Usage Data**: IP address, device fingerprints, browser details, and navigation metrics collected via cookies.

---

### 4. Lawful Bases for Processing (Section 30 & 32)
We only process your personal data where at least one of the following statutory grounds applies:
- **Consent (Section 32)**: Where you have granted explicit, freely given, and unbundled consent. You may withdraw consent at any time without detriment.
- **Contractual Performance**: To fulfill orders, deliver services, or process M-Pesa transactions requested by you.
- **Legal Obligation**: Compliance with Kenyan statutory enactments (including KRA tax filings, CBK AML regulations, and ODPC directives).
- **Legitimate Interests**: Enhancing platform security and preventing fraudulent activities.

---

### 5. Direct Marketing and Commercial Communications (Section 37)
In strict compliance with **Section 37 of the Data Protection Act**:
- We shall **never** use your personal data for direct marketing or promotional communications without prior affirmative consent.
- You have the absolute right to opt out of commercial communications at any time, at **zero financial cost**, by clicking the 'Unsubscribe' link in any email or sending an SMS opt-out to our shortcode.

---

### 6. Retention and Storage Limitation (Section 39)
Your personal data will not be retained longer than necessary:
- **Customer Transaction & Accounting Records**: Retained for seven (7) years to satisfy Kenyan tax and commercial statutory audits, after which records are irreversibly anonymized or securely shredded.
- **Account Profiles**: Retained for the active duration of your account and securely erased within thirty (30) days following account closure.
- **Customer Support Logs**: Purged after twelve (12) months.

---

### 7. Cross-Border Transfers Outside Kenya (Part VI, Sections 48 & 49)
Where personal data is stored on cloud infrastructure located outside the territory of Kenya (e.g. AWS, Microsoft Azure, Google Cloud):
- Transfers are executed only to jurisdictions with adequate legal protections verified by the ODPC, or protected by binding **Standard Contractual Clauses (SCCs)** incorporating Kenyan statutory data subject protections.
- Sensitive personal data is processed strictly in compliance with **Section 49** with prior ODPC notification.

---

### 8. Security Safeguards & 72-Hour Breach Notification (Section 43)
We employ bank-grade technical and organizational safeguards including AES-256 encryption at rest, TLS 1.3 transit encryption, strict role-based access control, and continuous vulnerability monitoring.

In the event of a personal data breach that presents a real risk of harm, **{name} shall notify the Data Commissioner within seventy-two (72) hours** pursuant to **Section 43(1) of the Act**, and inform affected Data Subjects without undue delay.

---

### 9. Your Statutory Data Subject Rights (Section 26 & Section 40)
Under Kenyan law, you possess enforceable rights regarding your personal data:
- **Right to be Informed**: Right to know how your data is collected and processed.
- **Right of Access**: Request a certified copy of your personal data held by us.
- **Right to Rectification**: Request correction of false, misleading, or outdated personal data.
- **Right to Erasure**: Request complete deletion of data where no lawful justification for retention exists.
- **Right to Object**: Object to processing based on legitimate interests or direct marketing.
- **Right to Data Portability**: Receive your data in a structured, commonly used, machine-readable format.
- **Automated Decision Protections (Section 40)**: Right not to be subjected to decisions based solely on automated processing.

To exercise any of these rights, email our DPO at: `dpo@{name.lower().replace(' ', '')}.co.ke`. We will resolve requests within fourteen (14) calendar days.

---

### 10. Lodging a Complaint with the Supervisory Authority (ODPC)
If you believe our processing of your personal data violates the Kenya Data Protection Act 2019, you have the statutory right under **Section 56** to lodge a formal complaint with:

**The Office of the Data Protection Commissioner (ODPC)**
- **Physical Address**: Britam Tower, 12th & 13th Floor, Hospital Road, Upper Hill, Nairobi
- **Postal Address**: P.O. Box 30920-00100, Nairobi, Kenya
- **Email**: `complaints@odpc.go.ke` / `info@odpc.go.ke`
- **Official Website**: [https://www.odpc.go.ke](https://www.odpc.go.ke)
"""

    executive_summary = (
        f"This regulatory compliance audit conducted for {name} identifies {len(violations)} non-compliant areas "
        f"under the Kenya Data Protection Act, 2019 and ODPC Regulations 2021. With an overall compliance score of "
        f"{overall_score}/100 and a risk rating of {risk_level}, the entity faces substantial exposure to regulatory "
        f"enforcement notices and statutory penalty notices under Section 63 (which permits fines up to KES 5,000,000 "
        f"or 1% of annual turnover). Immediate remediation of direct marketing opt-outs, statutory breach reporting protocols, "
        f"and ODPC supervisory authority disclosures is urgently required."
    )

    return {
        "risk_level": risk_level,
        "overall_score": overall_score,
        "executive_summary": executive_summary,
        "violations": violations,
        "compliant_policy_draft": compliant_policy.strip()
    }


def create_teaser_summary(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs the free teaser summary exposing ONLY the first violation in full,
    locking and blurring the remaining violations and full compliant rewrite behind M-Pesa paywall.
    """
    violations = full_report.get("violations", [])
    first_violation = violations[0] if violations else None
    
    # Locked violations summary (titles, severities previewed, details obscured)
    locked_violations_preview = []
    if len(violations) > 1:
        for idx, v in enumerate(violations[1:], start=2):
            locked_violations_preview.append({
                "id": v.get("id", idx),
                "title": v.get("title", "Statutory DPA Violation"),
                "severity": v.get("severity", "High"),
                "is_locked": True
            })

    return {
        "risk_level": full_report.get("risk_level", "High"),
        "overall_score": full_report.get("overall_score", 45),
        "total_violations": len(violations),
        "unlocked_first_violation": first_violation,
        "locked_violations_preview": locked_violations_preview,
        "locked_count": max(0, len(violations) - 1),
        "has_compliant_draft": bool(full_report.get("compliant_policy_draft"))
    }


def run_compliance_audit(document_text: str, target_url_or_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the full Kenyan ODPC compliance audit pipeline:
    1. Vector similarity search against legal_chunks using pgvector.
    2. Regulatory compliance evaluation via LLM (with heuristic fallback).
    3. Creation of teaser and full report structures.
    
    Args:
        document_text: Raw or scraped Privacy Policy text.
        target_url_or_name: Optional URL or business identifier.
        
    Returns:
        Dict containing:
            - "risk_level": "High" | "Medium" | "Low"
            - "teaser_summary": JSON-serializable teaser dict
            - "full_report": Complete JSON-serializable findings
            - "compliant_rewrite": Complete ODPC-compliant markdown policy
    """
    clean_text = document_text.strip()
    if len(clean_text) < 100:
        raise ValueError("Policy text is too short to audit (minimum 100 characters required).")

    entity_name = target_url_or_name or "The Organization"
    if entity_name.startswith("http"):
        parsed = urlparse(entity_name)
        entity_name = parsed.netloc.replace("www.", "").capitalize()

    # 1. Retrieve statutory legal chunks via pgvector
    legal_chunks = retrieve_relevant_legal_context(clean_text)

    # 2. Run LLM inference or heuristic evaluation
    full_report = None
    if LLM_API_KEY and LLM_PROVIDER in ["openai", "openai_compatible"]:
        try:
            full_report = _audit_with_llm(clean_text, legal_chunks, entity_name)
            logger.info("Successfully completed compliance audit with LLM provider '%s'.", LLM_PROVIDER)
        except Exception as llm_err:
            logger.warning("LLM evaluation failed or timed out (%s). Falling back to statutory heuristics.", llm_err)

    if not full_report:
        logger.info("Running deterministic statutory heuristic evaluation...")
        full_report = _audit_with_heuristics(clean_text, legal_chunks, entity_name)

    # Validate output schema
    risk_level = full_report.get("risk_level", "High")
    if risk_level not in ["High", "Medium", "Low"]:
        risk_level = "High"

    compliant_draft = full_report.get("compliant_policy_draft", "")
    teaser_summary = create_teaser_summary(full_report)

    return {
        "risk_level": risk_level,
        "teaser_summary": teaser_summary,
        "full_report": full_report,
        "compliant_rewrite": compliant_draft
    }
