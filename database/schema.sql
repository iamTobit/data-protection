-- ====================================================================
-- Kenyan Data Protection Act (ODPC) Compliance Auditor
-- Database Schema for Supabase / PostgreSQL with pgvector
-- ====================================================================

-- 1. Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Legal Chunks Table
-- Stores statutory sections, ODPC regulations, guidance notes, and enforcement rulings
-- Embedded using BAAI/bge-small-en-v1.5 (384 dimensions)
CREATE TABLE IF NOT EXISTS legal_chunks (
    id BIGSERIAL PRIMARY KEY,
    act_title VARCHAR(255) NOT NULL,              -- e.g. 'Data Protection Act 2019', 'Data Protection (General) Regulations 2021'
    section_ref VARCHAR(100),                     -- e.g. 'Section 25', 'Section 32', 'Regulation 14'
    citation VARCHAR(255) NOT NULL,               -- e.g. 'Kenya Data Protection Act, 2019, s. 25(1)(a)'
    content TEXT NOT NULL,                        -- Actual legal provision text and guidance
    embedding vector(384),                        -- 384-dimensional BGE embedding
    metadata JSONB DEFAULT '{}'::jsonb,           -- Additional metadata (topic tag, source url, page, etc.)
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast cosine similarity vector search
-- Using HNSW (Hierarchical Navigable Small World) with vector_cosine_ops
CREATE INDEX IF NOT EXISTS idx_legal_chunks_embedding 
ON legal_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Text search / reference lookup index
CREATE INDEX IF NOT EXISTS idx_legal_chunks_citation 
ON legal_chunks (citation);

CREATE INDEX IF NOT EXISTS idx_legal_chunks_act_title 
ON legal_chunks (act_title);

-- 3. Audit Reports Table
-- Stores compliance audit sessions, findings, teaser data, and M-Pesa payment status
CREATE TABLE IF NOT EXISTS audit_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_url_or_name VARCHAR(512),              -- Business website URL or entity name
    raw_input_text TEXT NOT NULL,                 -- Scraped privacy policy or raw submitted text
    risk_level VARCHAR(50) NOT NULL,              -- 'High', 'Medium', 'Low'
    teaser_summary JSONB NOT NULL DEFAULT '{}'::jsonb, -- Summary with only first violation revealed for free
    full_report JSONB NOT NULL DEFAULT '{}'::jsonb,    -- Complete audit findings, violations list, scores
    compliant_rewrite TEXT,                       -- Fully tailored ODPC-compliant replacement Privacy Policy
    is_paid BOOLEAN NOT NULL DEFAULT false,       -- Unlocks full report upon KES 1,000 M-Pesa confirmation
    mpesa_checkout_id VARCHAR(100),               -- Daraja CheckoutRequestID for tracking STK push
    mpesa_receipt_number VARCHAR(100),            -- Safaricom transaction code (e.g. 'SHG78291AA')
    phone_number VARCHAR(50),                     -- Phone number prompted for M-Pesa payment
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for rapid M-Pesa callback reconciliation
CREATE INDEX IF NOT EXISTS idx_audit_reports_checkout_id 
ON audit_reports (mpesa_checkout_id);

-- Index for ordering reports by date
CREATE INDEX IF NOT EXISTS idx_audit_reports_created_at 
ON audit_reports (created_at DESC);
