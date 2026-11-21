# 🇰🇪 Kenyan Data Protection (ODPC) Compliance Auditor SaaS

A production-ready AI compliance evaluation and SaaS web application tailored to the **Republic of Kenya Data Protection Act, 2019 (DPA 2019)** and the **Data Protection (General) Regulations, 2021** enforced by the **Office of the Data Protection Commissioner (ODPC)**.

---

## 🏛️ System Architecture

```
[ User Browser / Client ]
       │
       ▼
 [ Flask App (app.py) ] ──────────────────────┐
       │                                       │
       ├─► [ services/rag_auditor.py ]         ├─► [ services/mpesa.py ]
       │         │                             │         │
       │         ├─► Embeddings:               │         └─► Safaricom Daraja
       │         │   BAAI/bge-small-en-v1.5    │             Lipa Na M-Pesa
       │         │                             │             (STK Push KES 1,000)
       │         └─► pgvector Retrieval        │
       │                                       │
       ▼                                       ▼
 [ Supabase / PostgreSQL (psycopg v3) ] ◄──────┘
       ├── legal_chunks (vector(384) + HNSW)
       └── audit_reports (UUID, JSONB reports, is_paid)
```

### Key Technical Pillars:
1. **Python Flask Backend**: Production-ready, modular architecture with error handling, logging, and environment configuration.
2. **PostgreSQL / Supabase with pgvector**: Stores statutory chunks from the Kenya Data Protection Act 2019 and ODPC enforcement decisions embedded as **384-dimensional vectors** indexed with **HNSW** (`vector_cosine_ops`).
3. **Local Embedding Engine**: Uses `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384 dimensions) for normalized vector similarity.
4. **LLM Inference Layer**: Supports OpenAI-compatible API endpoints (OpenAI, Groq, Together, Ollama, vLLM) and Hugging Face, with a resilient rule-based statutory heuristic fallback.
5. **Safaricom Daraja Lipa Na M-Pesa**: Full STK Push integration (KES 1,000 unlock fee) with automated webhook callback reconciliation and client polling.
6. **Modern Frontend**: Responsive HTML5 + Tailwind CSS (via CDN) + Lucide icons + minimal vanilla JS.

---

## 📂 Project Structure

```
data-protection-app/
├── Pipfile                         # Pipenv dependency manifest
├── .env.example                    # Environment variable template
├── app.py                          # Flask application routes and entry point
├── database/
│   ├── schema.sql                  # PostgreSQL schema with pgvector extensions & tables
│   └── connection.py               # psycopg (v3) connection manager & pgvector registration
├── scripts/
│   └── ingest_laws.py              # CLI ingestion tool for web URLs, PDFs & foundational laws
├── services/
│   ├── rag_auditor.py              # Web scraper, embedding generator, RAG retrieval & audit engine
│   └── mpesa.py                    # Safaricom Daraja STK Push & Webhook callback processor
├── templates/
│   ├── base.html                   # Global HTML layout with Kenya national theme & Lucide icons
│   ├── index.html                  # Landing page & dual scanner (URL or Raw Text)
│   └── report.html                 # Audit findings, teaser preview, blurred paywall & M-Pesa modal
├── odpc_pdfs/                      # Directory for local ODPC rulings/guidance PDFs
└── README.md                       # Comprehensive setup and operations manual
```

---

## ⚡ Quick Start & Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- `pipenv` or `pip`
- A PostgreSQL database with `vector` extension enabled (e.g. Supabase or local PostgreSQL 15+)

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and configure your settings:
```ini
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=replace-with-a-secure-random-secret
PORT=5000

# Supabase or PostgreSQL Connection
DATABASE_URL=postgresql://postgres:[PASSWORD]@[HOST]:[PORT]/postgres

# LLM Inference (Optional: defaults to built-in heuristic auditor if key is not set)
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1

# Safaricom Daraja M-Pesa API
MPESA_ENVIRONMENT=sandbox
MPESA_CONSUMER_KEY=your_daraja_consumer_key
MPESA_CONSUMER_SECRET=your_daraja_consumer_secret
MPESA_SHORTCODE=174379
MPESA_PASSKEY=bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919
MPESA_CALLBACK_URL=https://your-ngrok-subdomain.ngrok-free.app/api/mpesa/callback
REPORT_UNLOCK_PRICE_KES=1000
```

---

### 3. Install Dependencies
Using `pipenv`:
```bash
pipenv install
pipenv shell
```
Or directly with `pip`:
```bash
pip install -r Pipfile
# or:
pip install flask "psycopg[binary,pool]" pgvector sentence-transformers requests beautifulsoup4 pypdf python-dotenv openai gunicorn
```

---

### 4. Database Setup & Legal Corpus Ingestion

#### Step A: Run Schema on Supabase / PostgreSQL
Run the SQL script `database/schema.sql` inside your Supabase SQL Editor or via `psql`:
```bash
psql $DATABASE_URL -f database/schema.sql
```
Alternatively, initialize it via the ingestion script:
```bash
python scripts/ingest_laws.py --init-db
```

#### Step B: Seed Foundational Kenyan Data Protection Laws
Ingest the embedded statutory corpus of the Kenya Data Protection Act 2019 and ODPC Regulations 2021 into `legal_chunks`:
```bash
python scripts/ingest_laws.py --seed-sample
```

#### Step C: (Optional) Ingest External URLs & Local PDFs
To scrape and chunk statutory text directly from a web URL:
```bash
python scripts/ingest_laws.py --url "https://kenyalaw.org/kl/fileadmin/pdfdownloads/Acts/2019/TheDataProtectionAct__No24of2019.pdf"
```

To ingest any local ODPC penalty notices or determination PDFs in `odpc_pdfs/`:
```bash
python scripts/ingest_laws.py --pdf-dir odpc_pdfs/
```

To execute all ingestion sources at once:
```bash
python scripts/ingest_laws.py --all
```

---

### 5. Running the Application

Start the Flask development server:
```bash
python app.py
```
Or with Gunicorn for production:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Open `http://localhost:5000` in your browser.

---

## 🧪 Testing the Compliance Scanner

1. **Option 1: Scan Live URL**
   - Enter a Kenyan business URL (e.g. `https://example.com`). The crawler will automatically inspect the page, locate the `/privacy` or `/terms` link, extract the plain text, and run the audit.
2. **Option 2: Paste Raw Policy Text**
   - Click the "Paste Policy / Contract Text" tab.
   - Use the **1-Click Preload Buttons** on the landing page:
     - **E-Commerce Demo (Missing Marketing Opt-Out)**: Tests Section 37 direct marketing violations.
     - **Digital Lending Demo (Critical ODPC Violations)**: Tests third-party contacts, contact list scraping, and retention breaches.
3. Click **"Run Kenyan ODPC Compliance Audit"**.
4. The system:
   - Computes query embeddings with `BAAI/bge-small-en-v1.5`.
   - Queries `legal_chunks` using pgvector cosine distance (`<=>`).
   - Evaluates the policy against Kenyan statutory criteria.
   - Saves the report in `audit_reports` with `is_paid = false`.
   - Redirects to `/report/<audit_id>`.

---

## 💰 M-Pesa Paywall & Unlocking

The report view reveals:
- **Free Preview**: Overall Risk Level badge (High, Medium, Low), compliance index score, and **Violation #1 fully unlocked** with exact statutory citation (e.g. *Kenya Data Protection Act 2019, Section 37*).
- **Paywalled Content**: Violations #2..N and the complete, publication-ready replacement Privacy Policy draft are visually blurred with a frosted glass backdrop.

### Testing Payment Unlocking:
1. Click **"Unlock Full Report & Compliant Policy (KES 1,000 via M-Pesa)"**.
2. Enter a Safaricom phone number (e.g., `0712345678`).
3. Click **"Send M-Pesa STK Push"**:
   - In production with Daraja credentials: An STK PIN prompt appears on the phone. The frontend polls `/api/report/<audit_id>/status` every 2 seconds and automatically unlocks the page when Safaricom posts to `/api/mpesa/callback`.
   - In developer sandbox or local testing: Click the **"Developer Sandbox: Simulate Immediate Unlock"** button in the modal to instantly verify the unlocked view and test downloading/copying the compliant policy draft!

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page and scanner form |
| `POST` | `/api/audit` | Accepts `{scan_type, target_url, policy_text}`, runs RAG audit, returns `{success, audit_id, redirect_url}` |
| `GET` | `/report/<audit_id>` | HTML view of audit report (Free preview or full unlocked) |
| `POST` | `/api/pay/<audit_id>` | Triggers Daraja M-Pesa STK Push for KES 1,000 |
| `POST` | `/api/mpesa/callback` | Safaricom Daraja Webhook callback receiver (`ResultCode == 0` unlocks report) |
| `GET` | `/api/report/<audit_id>/status` | Polling endpoint returning `{is_paid: bool, risk_level: str}` |
| `POST` | `/api/pay/simulate-sandbox/<audit_id>` | Developer helper to simulate payment confirmation in test environments |
| `GET` | `/health` | Health check verifying database and vector connectivity |

---

## ⚖️ Kenyan Data Protection Act (ODPC) Summary Matrix

| Section | Topic | Requirement Under Kenyan Law |
|---|---|---|
| **Section 25** | Principles | Lawfulness, fairness, transparency, purpose limitation, data minimisation, storage limitation. |
| **Section 26** | Data Subject Rights | Right to be informed, right of access, right of rectification, right to object, right to erasure. |
| **Section 30** | Lawful Basis | Consent, contract performance, legal obligation, vital interests, legitimate interests. |
| **Section 32** | Consent | Freely given, specific, informed. Controller bears the burden of proof. Simple withdrawal. |
| **Section 37** | Direct Marketing | Commercial communication prohibited without prior consent. Zero-cost opt-out facility mandatory. |
| **Section 39** | Retention | Storage limitation. Vague "as long as necessary" phrases violate Kenyan law. Explicit schedules required. |
| **Section 43** | Breach Notification | Mandatory notification of personal data breaches to the ODPC within **72 hours**. |
| **Section 48** | Cross-Border Transfer | Transfers outside Kenya require proof of appropriate safeguards, adequacy, or explicit consent. |
| **Section 63** | Penalties & Fines | Statutory penalties up to **KES 5,000,000** or 1% of annual turnover, plus director liability. |

---

## 🛡️ License
MIT License. Developed for Kenyan organizations, compliance officers, and legal teams seeking proactive ODPC audit readiness.
# data-protection
