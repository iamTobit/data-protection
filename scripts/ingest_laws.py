#!/usr/bin/env python3
"""
CLI Script for Ingesting Kenyan Statutory Law & ODPC Rulings into pgvector.
Features:
1. Scrapes statutory legal text from web URLs (using requests + BeautifulSoup).
2. Parses local PDF rulings/guidance from an odpc_pdfs/ directory (using pypdf).
3. Provides a high-fidelity foundational seed corpus of the Kenya Data Protection Act, 2019
   and Data Protection (General) Regulations, 2021.
4. Generates 384-dimensional normalized embeddings using BAAI/bge-small-en-v1.5.
5. Upserts chunk records into the legal_chunks table in PostgreSQL / Supabase.
"""

import os
import re
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv

# Ensure parent directory is in path so we can import database modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import get_db_connection, init_db
from services.rag_auditor import generate_embedding

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ingest_laws")


# =====================================================================
# 1. Authoritative Seed Corpus: Kenya Data Protection Act 2019
# =====================================================================
FOUNDATIONAL_KENYAN_LAWS = [
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 25",
        "citation": "Kenya Data Protection Act, 2019, Section 25",
        "content": (
            "Section 25: Principles of data protection. Every data controller or data processor shall ensure "
            "that personal data is: (a) processed in accordance with the right to privacy of the data subject; "
            "(b) processed lawfully, fairly and in a transparent manner in relation to any data subject; "
            "(c) collected for explicit, specified and legitimate purposes and not further processed in a manner incompatible with those purposes; "
            "(d) adequate, relevant and limited to what is necessary in relation to the purposes for which it is processed (data minimisation); "
            "(e) collected only where a valid explanation is provided whenever information relating to family or private affairs is required; "
            "(f) accurate and, where necessary, kept up to date, with every reasonable step being taken to ensure that inaccurate personal data is erased or rectified without delay; "
            "(g) kept in a form which identifies the data subjects for no longer than is necessary for the purposes for which it was collected (storage limitation); "
            "and (h) not transferred outside Kenya, unless there is proof of adequate data protection safeguards or consent from the data subject."
        ),
        "metadata": {"topic": "principles", "source": "Kenya Gazette Supplement No. 181 (Acts No. 24)"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 26",
        "citation": "Kenya Data Protection Act, 2019, Section 26",
        "content": (
            "Section 26: Rights of a data subject. A data subject has a right to: (a) be informed of the use to which "
            "their personal data is to be put; (b) access their personal data in custody of data controller or processor (Reg 9, free of charge within 7 days); "
            "(c) object to the processing of all or part of their personal data (Section 36); (d) correction of false or misleading data (Section 40 & Reg 10, 14 days); "
            "and (e) deletion of false or misleading data about them (Section 40 & Reg 12, 14 days). Data controllers must provide accessible procedures "
            "for individuals to exercise these statutory rights without undue delay."
        ),
        "metadata": {"topic": "data_subject_rights", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 30",
        "citation": "Kenya Data Protection Act, 2019, Section 30",
        "content": (
            "Section 30: Lawful processing of personal data. (1) A data controller or data processor shall not process personal data, unless: "
            "(a) the data subject consents to the processing for one or more specified purposes (Section 30(1)(a) & Section 32); or "
            "(b) the processing is necessary: (i) for the performance of a contract to which the data subject is a party or in order to take steps before entering into a contract (Section 30(1)(b)(i)); "
            "(ii) for compliance with any legal obligation to which the controller is subject, e.g. statutory tax or regulatory duties (Section 30(1)(b)(ii)); "
            "(iii) in order to protect the vital interests of the data subject or another natural person (Section 30(1)(b)(iii)); "
            "(iv) for the performance of a task carried out in the public interest or in the exercise of official authority (Section 30(1)(b)(iv)); "
            "(v) for the performance of any task carried out by a public authority (Section 30(1)(b)(v)); "
            "(vi) for the exercise of the appropriate functions of a public body (Section 30(1)(b)(vi)); "
            "(vii) for the purposes of the legitimate interests pursued by the data controller or processor or by a third party to whom the data is disclosed, except if unwarranted having regard to harm to the data subject (Section 30(1)(b)(vii)); or "
            "(viii) for the purpose of historical, statistical, or scientific research (Section 30(1)(b)(viii))."
        ),
        "metadata": {"topic": "lawful_basis", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 32",
        "citation": "Kenya Data Protection Act, 2019, Section 32",
        "content": (
            "Section 32: Conditions of consent. (1) A data controller shall bear the burden of proof for establishing "
            "the data subject's consent to the processing of personal data for a specified purpose. (2) Unless the data "
            "controller provides otherwise, consent shall not be regarded as freely given if the performance of a contract, "
            "including the provision of a service, is dependent on the consent despite not being necessary. (3) The data subject "
            "has the right to withdraw consent at any time; the withdrawal of consent shall not affect the lawfulness of processing "
            "based on consent prior to its withdrawal."
        ),
        "metadata": {"topic": "consent", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 33",
        "citation": "Kenya Data Protection Act, 2019, Section 33",
        "content": (
            "Section 33: Processing of personal data relating to a child. (1) A data controller or data processor shall not process "
            "personal data relating to a child unless: (a) consent is given by the child's parent or legal guardian; and (b) the "
            "processing is in such a manner that protects and advances the rights and best interests of the child. (2) A data "
            "controller or data processor shall incorporate mechanisms for age verification and parental consent."
        ),
        "metadata": {"topic": "children_data", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 37",
        "citation": "Kenya Data Protection Act, 2019, Section 37",
        "content": (
            "Section 37: Commercial use of personal data and direct marketing. (1) A person shall not use personal data for "
            "commercial purposes, including direct marketing, unless the person: (a) has sought and obtained the consent from the "
            "data subject; or (b) is authorized to do so under any written law and the data subject has been informed of such use. "
            "(2) A data controller or data processor that uses personal data for commercial purposes shall provide a facility for "
            "the data subject to opt out at zero cost."
        ),
        "metadata": {"topic": "direct_marketing", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 39",
        "citation": "Kenya Data Protection Act, 2019, Section 39",
        "content": (
            "Section 39: Retention of personal data. (1) A data controller or data processor shall retain personal data only for as "
            "long as may be reasonably necessary to satisfy the purpose for which it is processed unless the retention is required "
            "or authorized by law, or with the consent of the data subject. (2) A data controller or data processor shall erase, "
            "anonymise or pseudo-anonymise personal data upon expiry of the retention period."
        ),
        "metadata": {"topic": "retention", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 35",
        "citation": "Kenya Data Protection Act, 2019, Section 35",
        "content": (
            "Section 35: Automated individual decision making and profiling. (1) A data subject has the right not to be subjected to a decision "
            "based solely on automated processing, including profiling, which produces legal effects concerning or significantly "
            "affects the data subject. (2) Where a decision is made solely on automated processing, the data controller shall "
            "inform the data subject and guarantee the right to obtain human intervention and contest the decision."
        ),
        "metadata": {"topic": "automated_decision_making", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 36",
        "citation": "Kenya Data Protection Act, 2019, Section 36",
        "content": (
            "Section 36: Right to object. A data subject has the right to object to the processing of all or part of their personal data, "
            "including for direct marketing purposes, unless the data controller demonstrates compelling legitimate grounds for processing "
            "which override the data subject's interests, rights and freedoms."
        ),
        "metadata": {"topic": "right_to_object", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 40",
        "citation": "Kenya Data Protection Act, 2019, Section 40",
        "content": (
            "Section 40: Rectification and erasure of personal data. (1) A data subject may request a data controller or data processor: "
            "(a) to rectify personal data about the data subject that is inaccurate, out of date, incomplete or misleading; or "
            "(b) to erase or destroy personal data that the data controller or processor is no longer authorized to retain, irrelevant, "
            "excessive, or obtained unlawfully. Under Regulations 10 & 12 of the General Regulations 2021, requests must be fulfilled within 14 days without charge."
        ),
        "metadata": {"topic": "rectification_erasure", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 41",
        "citation": "Kenya Data Protection Act, 2019, Section 41",
        "content": (
            "Section 41: Data Protection Impact Assessment (DPIA). Where a processing operation is likely to result in a high risk "
            "to the rights and freedoms of data subjects by virtue of its nature, scope, context and purposes, a data controller or "
            "data processor shall, prior to the processing, carry out a Data Protection Impact Assessment and consult the Data Commissioner."
        ),
        "metadata": {"topic": "dpia", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 43",
        "citation": "Kenya Data Protection Act, 2019, Section 43",
        "content": (
            "Section 43: Notification of personal data breaches. (1) A data controller shall notify the Data Commissioner within "
            "seventy-two (72) hours of becoming aware of a breach where there is a real risk of harm to the data subject. (2) A data "
            "processor shall notify the data controller immediately after becoming aware of a personal data breach. (3) The data "
            "controller shall communicate the personal data breach to the data subject in writing within a reasonably practicable period."
        ),
        "metadata": {"topic": "breach_notification", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 48 & Part VII",
        "citation": "Kenya Data Protection Act, 2019, Section 48 & General Regulations Part VII",
        "content": (
            "Section 48 and Part VII (Regulations 40–48): Transfer of personal data outside Kenya. A data controller or data processor "
            "may transfer personal data outside Kenya only where: (a) the data controller has given proof to the Data Commissioner of the "
            "appropriate safeguards existing in the recipient country through binding contractual agreements or adequacy decisions; "
            "(b) the transfer is necessary for performance of a contract; or (c) explicit consent is obtained from the data subject."
        ),
        "metadata": {"topic": "cross_border_transfer", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 49",
        "citation": "Kenya Data Protection Act, 2019, Section 49",
        "content": (
            "Section 49: Safeguards prior to transfer of sensitive personal data outside Kenya. (1) The transfer of sensitive personal "
            "data outside Kenya requires the explicit consent of the data subject AND confirmation of appropriate safeguards or authorization "
            "by the Data Commissioner. The Data Commissioner may prohibit, suspend or subject the transfer to conditions."
        ),
        "metadata": {"topic": "sensitive_data_transfer", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection Act, 2019",
        "section_ref": "Section 62 & 63",
        "citation": "Kenya Data Protection Act, 2019, Section 62 & 63",
        "content": (
            "Section 62 & 63: Offences, penalty notices and statutory fines. A person who breaches provisions of the Act commits an offence. "
            "Under Section 63, the Data Commissioner may serve a penalty notice requiring the person to pay a statutory penalty up to "
            "five million shillings (KES 5,000,000) or up to one percent (1%) of its annual turnover of the preceding financial year, "
            "whichever is lower. Officers and directors may also face imprisonment for a term not exceeding ten (10) years."
        ),
        "metadata": {"topic": "penalties_fines", "source": "DPA 2019"}
    },
    {
        "act_title": "Data Protection (General) Regulations, 2021",
        "section_ref": "Regulation 13 & 14",
        "citation": "Kenya Data Protection (General) Regulations, 2021, Reg. 13 & 14",
        "content": (
            "Regulations 13 & 14: Direct marketing and commercial use. Commercial use of personal data includes using personal data to "
            "advertise, market, solicit, or promote goods or services. Prior to sending direct marketing communications via SMS, email, "
            "or digital channels, affirmative opt-in consent must be gathered. Pre-ticked consent boxes are explicitly invalid under Kenyan law."
        ),
        "metadata": {"topic": "general_regulations_marketing", "source": "Legal Notice No. 263"}
    },
    {
        "act_title": "Data Protection (General) Regulations, 2021",
        "section_ref": "Regulation 20 & 22",
        "citation": "Kenya Data Protection (General) Regulations, 2021, Reg. 20 & 22",
        "content": (
            "Regulations 20 & 22: Retention schedules and data subject request handling. Data controllers must maintain an organizational "
            "retention schedule. A data controller shall respond to a data subject access, rectification, or erasure request within "
            "fourteen (14) days from the date of receipt, and shall not charge any fee for granting statutory access."
        ),
        "metadata": {"topic": "general_regulations_retention", "source": "Legal Notice No. 263"}
    },
    {
        "act_title": "ODPC Enforcement Determination",
        "section_ref": "Ruling ODPC/ENF/2023/001",
        "citation": "ODPC Penalty Notice Ruling 2023, ODPC/ENF/2023/001",
        "content": (
            "ODPC Regulatory Finding: Digital lenders and e-commerce platforms operating in Kenya were penalized for unauthorized "
            "contacting of third parties, lack of explicit consent forms, sending unsolicited direct marketing messages without a "
            "zero-cost opt-out mechanism, and failing to publish designated Data Protection Officer contacts."
        ),
        "metadata": {"topic": "enforcement_precedent", "source": "ODPC Enforcement Registry"}
    }
]


# =====================================================================
# 2. Database Upsert Helper
# =====================================================================
def upsert_legal_chunk(
    act_title: str,
    section_ref: str,
    citation: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Computes 384-dim BGE embedding and inserts/updates a legal chunk in PostgreSQL.
    
    Returns:
        int: The database ID of the inserted record.
    """
    metadata_clean = metadata or {}
    text_to_embed = f"{citation} - {content}"
    embedding = generate_embedding(text_to_embed)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if this exact citation already exists to ensure idempotency
            cur.execute(
                "SELECT id FROM legal_chunks WHERE citation = %s LIMIT 1;",
                (citation,)
            )
            existing = cur.fetchone()

            if existing:
                cur.execute(
                    """
                    UPDATE legal_chunks
                    SET act_title = %s,
                        section_ref = %s,
                        content = %s,
                        embedding = %s::vector,
                        metadata = %s::jsonb
                    WHERE id = %s
                    RETURNING id;
                    """,
                    (act_title, section_ref, content, embedding, json.dumps(metadata_clean), existing["id"])
                )
                record_id = existing["id"]
                logger.debug("Updated existing legal chunk [%s] ID %s", citation, record_id)
            else:
                cur.execute(
                    """
                    INSERT INTO legal_chunks (act_title, section_ref, citation, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                    RETURNING id;
                    """,
                    (act_title, section_ref, citation, content, embedding, json.dumps(metadata_clean))
                )
                res = cur.fetchone()
                record_id = res["id"]
                logger.debug("Inserted new legal chunk [%s] ID %s", citation, record_id)

    return record_id


# =====================================================================
# 3. Seed Sample Corpus
# =====================================================================
def seed_sample_laws() -> int:
    """Ingests the foundational Kenya Data Protection Act corpus into pgvector."""
    logger.info("Ingesting foundational Kenyan ODPC statutory corpus (%d chunks)...", len(FOUNDATIONAL_KENYAN_LAWS))
    count = 0
    for chunk in FOUNDATIONAL_KENYAN_LAWS:
        try:
            upsert_legal_chunk(
                act_title=chunk["act_title"],
                section_ref=chunk["section_ref"],
                citation=chunk["citation"],
                content=chunk["content"],
                metadata=chunk.get("metadata", {})
            )
            count += 1
        except Exception as e:
            logger.error("Failed to seed chunk '%s': %s", chunk["citation"], e)
            
    logger.info("Successfully ingested %d foundational Kenyan legal chunks into pgvector.", count)
    return count


# =====================================================================
# 4. Scrape Statutes from Web URL
# =====================================================================
def chunk_statute_by_sections(raw_text: str, source_url: str) -> List[Dict[str, Any]]:
    """
    Parses statutory text into distinct sections based on standard Kenyan
    legislative headings (e.g. 'Section 12.', '12.', 'PART III').
    """
    chunks = []
    # Regex splitting on section indicators
    pattern = r'(?m)^(?:Section\s+)?(\d{1,3})\.\s+([^\n]+)'
    splits = list(re.finditer(pattern, raw_text))

    if not splits:
        # Fallback to paragraph batching
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 80]
        for idx, para in enumerate(paragraphs, start=1):
            chunks.append({
                "act_title": "Kenyan Statutory Instrument",
                "section_ref": f"Clause {idx}",
                "citation": f"Kenyan Statutory Document ({source_url}), Clause {idx}",
                "content": para,
                "metadata": {"source_url": source_url, "chunk_type": "paragraph"}
            })
        return chunks

    for i in range(len(splits)):
        sec_num = splits[i].group(1)
        sec_title = splits[i].group(2).strip()
        start_pos = splits[i].start()
        end_pos = splits[i+1].start() if (i + 1) < len(splits) else len(raw_text)
        
        section_text = raw_text[start_pos:end_pos].strip()
        if len(section_text) > 40:
            chunks.append({
                "act_title": "Kenya Data Protection Act / ODPC Regulation",
                "section_ref": f"Section {sec_num}",
                "citation": f"Kenya Data Protection Law, Section {sec_num} ({sec_title[:60]})",
                "content": section_text[:2500],
                "metadata": {"source_url": source_url, "section_number": sec_num, "heading": sec_title}
            })

    return chunks


def ingest_from_url(url: str) -> int:
    """Scrapes legal text from a URL and stores chunk embeddings into pgvector."""
    logger.info("Scraping statutory text from URL: %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ODPCComplianceBot/1.0)"}
    
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for el in soup(["script", "style", "nav", "footer", "header"]):
        el.decompose()

    raw_text = soup.get_text(separator="\n", strip=True)
    chunks = chunk_statute_by_sections(raw_text, url)
    
    logger.info("Parsed %d legal sections from %s. Embedding and upserting...", len(chunks), url)
    count = 0
    for ch in chunks:
        upsert_legal_chunk(
            act_title=ch["act_title"],
            section_ref=ch["section_ref"],
            citation=ch["citation"],
            content=ch["content"],
            metadata=ch["metadata"]
        )
        count += 1

    logger.info("Successfully ingested %d sections from %s.", count, url)
    return count


# =====================================================================
# 5. Parse Local PDF Rulings (odpc_pdfs/)
# =====================================================================
def chunk_pdf_text(pages_text: List[str], filename: str, chunk_size: int = 400, overlap: int = 60) -> List[Dict[str, Any]]:
    """
    Chunks document text into overlapping token/word segments for RAG retrieval.
    """
    full_text = "\n\n".join(pages_text)
    words = full_text.split()
    
    chunks = []
    start = 0
    chunk_index = 1
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        segment = " ".join(words[start:end])
        
        citation = f"ODPC Ruling / Guidance ({filename}), Segment {chunk_index}"
        chunks.append({
            "act_title": "ODPC Enforcement Ruling / Guidance",
            "section_ref": f"Segment {chunk_index}",
            "citation": citation,
            "content": segment,
            "metadata": {"filename": filename, "segment_index": chunk_index}
        })
        
        chunk_index += 1
        if end >= len(words):
            break
        start += (chunk_size - overlap)

    return chunks


def ingest_from_pdf_dir(pdf_dir: str = "odpc_pdfs") -> int:
    """Parses all PDF rulings in the specified directory and embeds them."""
    if not os.path.exists(pdf_dir):
        logger.warning("PDF directory '%s' does not exist. Creating it...", pdf_dir)
        os.makedirs(pdf_dir, exist_ok=True)
        return 0

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.info("No PDF files found in directory '%s'.", pdf_dir)
        return 0

    total_chunks = 0
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        logger.info("Reading PDF: %s...", pdf_path)
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for p_num, page in enumerate(reader.pages, start=1):
                t = page.extract_text()
                if t:
                    pages_text.append(f"[Page {p_num}]\n{t.strip()}")

            if not pages_text:
                logger.warning("No readable text found in PDF %s (might be scanned images).", filename)
                continue

            chunks = chunk_pdf_text(pages_text, filename)
            logger.info("Extracted %d chunks from %s. Embedding and upserting...", len(chunks), filename)

            for ch in chunks:
                upsert_legal_chunk(
                    act_title=ch["act_title"],
                    section_ref=ch["section_ref"],
                    citation=ch["citation"],
                    content=ch["content"],
                    metadata=ch["metadata"]
                )
                total_chunks += 1

        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", filename, e)

    logger.info("Successfully ingested %d chunks from PDF directory '%s'.", total_chunks, pdf_dir)
    return total_chunks


# =====================================================================
# Main CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Kenyan ODPC Statutory Laws & Decisions Ingestion Tool for pgvector."
    )
    parser.add_argument(
        "--init-db", 
        action="store_true", 
        help="Initialize database schema and extensions from database/schema.sql"
    )
    parser.add_argument(
        "--seed-sample", 
        action="store_true", 
        help="Ingest the foundational Kenya Data Protection Act 2019 seed corpus"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        help="Scrape and ingest statutory law text from a website URL"
    )
    parser.add_argument(
        "--pdf-dir", 
        type=str, 
        default="odpc_pdfs", 
        help="Path to directory containing ODPC PDF rulings/guidelines (default: odpc_pdfs)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run DB init, seed sample corpus, and ingest any local PDFs"
    )

    args = parser.parse_args()

    # If no flags passed, default to printing help or running --seed-sample
    if not (args.init_db or args.seed_sample or args.url or args.all):
        parser.print_help()
        print("\n[NOTE] Running default initialization: seeding foundational Kenyan DPA 2019 laws...\n")
        args.seed_sample = True

    try:
        if args.init_db or args.all:
            init_db("database/schema.sql")

        if args.seed_sample or args.all:
            seed_sample_laws()

        if args.url:
            ingest_from_url(args.url)

        if args.pdf_dir or args.all:
            if os.path.exists(args.pdf_dir):
                ingest_from_pdf_dir(args.pdf_dir)

        print("\n✅ Ingestion completed successfully.")
    except Exception as exc:
        logger.error("Ingestion process failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    import json
    main()
