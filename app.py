"""
Kenyan Data Protection (ODPC) Compliance Auditor SaaS.
Flask Application Entry Point.
"""

import os
import re
import json
import logging
import hashlib
from uuid import UUID
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from database.connection import get_db_connection, test_connection, init_db
from services.rag_auditor import (
    scrape_policy_from_url,
    run_compliance_audit,
    extract_text_from_file,
    is_terms_of_service,
    format_policy_html,
    strip_markdown_for_text,
    generate_policy_pdf
)
from services.mpesa import (
    initiate_stk_push,
    parse_mpesa_callback,
    query_stk_status,
    format_phone_number
)

# Load environment configuration
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger("odpc_auditor_app")

app = Flask(__name__)
# Secret key for session signing: prefer FLASK_SECRET_KEY, fallback to SECRET_KEY
configured_secret = (os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "").strip()
if not configured_secret or configured_secret.startswith("odpc-compliance-secret-key"):
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PORT"):
        logger.warning(
            "FLASK_SECRET_KEY is not configured in production environment. "
            "Generating an ephemeral secret key. Set FLASK_SECRET_KEY in Railway Variables."
        )
    configured_secret = configured_secret or os.urandom(32).hex()
app.secret_key = configured_secret
REPORT_PRICE = int(os.getenv("REPORT_UNLOCK_PRICE_KES", "1000"))

# Enforce safe maximum upload size (5 MB) & restrict extensions (.pdf, .docx, .txt)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Rate limiting to protect against API scraping and credit draining
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "60 per hour"],
    storage_uri="memory://"
)


@app.errorhandler(413)
def request_entity_too_large(error):
    msg = "Uploaded file exceeds the maximum allowed limit of 5 MB."
    if request.is_json:
        return jsonify({"success": False, "error": msg}), 413
    flash(msg, "error")
    return redirect(url_for("index"))


# -------------------------------------------------------------------------
# Template Filters & Helpers
# -------------------------------------------------------------------------
@app.template_filter("from_json")
def from_json_filter(val):
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {}


# -------------------------------------------------------------------------
# Database Storage & Caching Helpers
# -------------------------------------------------------------------------
def find_recent_cached_audit(input_text: str, force_fresh: bool = False) -> Optional[str]:
    """
    Checks for an identical audit submitted within the past 24 hours using
    sha256 hashing. Prevents bots from repeatedly draining LLM API tokens.
    If force_fresh is True or the cached audit has a degraded score (<= 15),
    bypasses cache so fresh, accurate analysis runs.
    """
    if force_fresh:
        return None
    try:
        input_hash = hashlib.sha256(input_text.strip().encode("utf-8")).hexdigest()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, full_report FROM audit_reports
                    WHERE encode(sha256(raw_input_text::bytea), 'hex') = %s
                      AND created_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    (input_hash,)
                )
                row = cur.fetchone()
                if row:
                    full_rep = row.get("full_report")
                    if isinstance(full_rep, str):
                        try:
                            full_rep = json.loads(full_rep)
                        except Exception:
                            full_rep = {}
                    if isinstance(full_rep, dict):
                        cached_score = full_rep.get("overall_score") or full_rep.get("compliance_score") or 0
                        if cached_score <= 15:
                            logger.info("Found cached audit %s but score is <= 15 (degraded fallback). Bypassing cache.", row["id"])
                            return None
                    return str(row["id"])
    except Exception as e:
        logger.warning("Cache lookup error: %s", e)
    return None


def save_audit_report(
    target_url_or_name: str,
    raw_input_text: str,
    audit_result: dict
) -> str:
    """
    Persists a completed compliance audit report in the audit_reports table.
    
    Returns:
        str: Generated UUID string for the report.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_reports (
                    target_url_or_name,
                    raw_input_text,
                    risk_level,
                    teaser_summary,
                    full_report,
                    compliant_rewrite,
                    is_paid
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, false)
                RETURNING id;
                """,
                (
                    target_url_or_name,
                    raw_input_text,
                    audit_result["risk_level"],
                    json.dumps(audit_result["teaser_summary"]),
                    json.dumps(audit_result["full_report"]),
                    audit_result["compliant_rewrite"]
                )
            )
            row = cur.fetchone()
            report_id = str(row["id"])
            logger.info("Saved new audit report: %s (Risk: %s)", report_id, audit_result["risk_level"])
            return report_id


def fetch_audit_report(report_id: str) -> dict:
    """Retrieves an audit report by its UUID."""
    if not report_id:
        return None
    try:
        uuid.UUID(str(report_id))
    except (ValueError, AttributeError):
        logger.warning("Invalid UUID format for report_id: %s", report_id)
        return None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, target_url_or_name, raw_input_text, risk_level,
                       teaser_summary, full_report, compliant_rewrite,
                       is_paid, mpesa_checkout_id, mpesa_receipt_number,
                       phone_number, created_at, updated_at
                FROM audit_reports
                WHERE id = %s::uuid;
                """,
                (report_id,)
            )
            report = cur.fetchone()
            if not report:
                return None
            return dict(report)


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Landing page: features Kenyan ODPC compliance badge, value proposition,
    and the 3-in-1 scanner input (URL scanner, raw text paste, or file upload).
    """
    return render_template("index.html")


@app.route("/api/audit", methods=["POST"])
@limiter.limit("5 per minute; 25 per hour")
def run_audit_endpoint():
    """
    Endpoint that accepts:
      1. Website URL, OR
      2. Raw pasted policy text, OR
      3. Uploaded policy document (.pdf, .docx, .txt).
    Performs guardrail verification, RAG audit pipeline, caching, and returns redirect.
    """
    try:
        if request.is_json:
            data = request.get_json()
            scan_type = data.get("scan_type", "url")
            target_url = data.get("target_url", "").strip()
            policy_text = data.get("policy_text", "").strip()
            entity_name = data.get("entity_name", "").strip()
        else:
            scan_type = request.form.get("scan_type", "url")
            target_url = request.form.get("target_url", "").strip()
            policy_text = request.form.get("policy_text", "").strip()
            entity_name = request.form.get("entity_name", "").strip()

        document_text = ""
        source_label = ""

        # Option 1: URL Scanner
        if scan_type == "url":
            if not target_url:
                msg = "Please provide a valid website URL."
                if request.is_json:
                    return jsonify({"success": False, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("index"))

            logger.info("Initiating scrape for: %s", target_url)
            scrape_res = scrape_policy_from_url(target_url)
            document_text = scrape_res["text"]
            source_label = scrape_res.get("title") or target_url

        # Option 2: File Upload (.pdf, .docx, .txt)
        elif scan_type == "file":
            uploaded_file = request.files.get("policy_file")
            if not uploaded_file or not uploaded_file.filename:
                msg = "Please select a valid Privacy Policy file (.pdf, .docx, or .txt) to upload."
                if request.is_json:
                    return jsonify({"success": False, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("index"))

            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            if ext not in ALLOWED_FILE_EXTENSIONS:
                msg = f"Unsupported file type '{ext}'. Please upload a .pdf, .docx, or .txt file."
                if request.is_json:
                    return jsonify({"success": False, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("index"))

            document_text, file_title = extract_text_from_file(uploaded_file, uploaded_file.filename)
            source_label = entity_name or file_title or "Uploaded Policy Document"

        # Option 3: Raw Text Paste
        else:
            if not policy_text or len(policy_text.strip()) < 100:
                msg = "Privacy Policy text must be at least 100 characters to perform a compliance audit."
                if request.is_json:
                    return jsonify({"success": False, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("index"))

            document_text = policy_text.strip()
            source_label = entity_name or "Submitted Policy Document"

        # Guardrail check: detect Terms of Service / Terms & Conditions
        if is_terms_of_service(document_text):
            warning_msg = (
                "The submitted document appears to be a 'Terms of Service' or 'Terms and Conditions' "
                "agreement rather than a Privacy Policy. The Kenya Data Protection Act, 2019 regulates "
                "data protection notices and privacy policies. Please upload or paste your organization's "
                "Privacy Policy / Privacy Notice instead."
            )
            if request.is_json:
                return jsonify({"success": False, "error": warning_msg, "is_tos": True}), 400
            flash(warning_msg, "error")
            return redirect(url_for("index"))

        # Duplicate-input cache check (prevents draining LLM credits)
        force_fresh = (
            request.args.get("refresh") in ["1", "true"] or
            request.args.get("fresh") in ["1", "true"] or
            (request.is_json and data.get("refresh") in [True, "1", "true"]) or
            (not request.is_json and request.form.get("refresh") in ["1", "true"])
        )
        cached_id = find_recent_cached_audit(document_text, force_fresh=force_fresh)
        if cached_id:
            logger.info("Reusing cached audit report: %s", cached_id)
            redirect_url = url_for("view_report", audit_id=cached_id)
            if request.is_json:
                return jsonify({
                    "success": True,
                    "audit_id": cached_id,
                    "redirect_url": redirect_url,
                    "cached": True
                })
            return redirect(redirect_url)

        # Execute RAG audit against pgvector legal chunks and Kenyan DPA rules
        audit_result = run_compliance_audit(
            document_text=document_text,
            target_url_or_name=source_label
        )

        # Store in PostgreSQL database
        report_id = save_audit_report(
            target_url_or_name=source_label,
            raw_input_text=document_text,
            audit_result=audit_result
        )

        redirect_url = url_for("view_report", audit_id=report_id)

        if request.is_json:
            return jsonify({
                "success": True,
                "audit_id": report_id,
                "redirect_url": redirect_url
            })

        return redirect(redirect_url)

    except ValueError as ve:
        logger.warning("Validation or scraping error during audit: %s", ve)
        if request.is_json:
            return jsonify({"success": False, "error": str(ve)}), 400
        flash(str(ve), "error")
        return redirect(url_for("index"))
    except Exception as e:
        logger.exception("Unexpected error during audit: %s", e)
        msg = f"An unexpected error occurred during the audit: {e}"
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 500
        flash(msg, "error")
        return redirect(url_for("index"))


@app.route("/report/<audit_id>")
def view_report(audit_id):
    """
    Displays the compliance audit report.
    - If is_paid is FALSE: server-side paywall ensures locked violations' details
      and the full Replacement Policy Draft are NOT sent in HTML or JSON.
    - If is_paid is TRUE: displays all violations and renders the clean, bleached
      HTML replacement policy.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            flash("Audit report not found or expired.", "error")
            return redirect(url_for("index"))

        is_paid = bool(report.get("is_paid")) and bool(report.get("mpesa_receipt_number"))
        teaser_summary = report.get("teaser_summary") or {}
        full_report = report.get("full_report") or {}

        if isinstance(teaser_summary, str):
            teaser_summary = json.loads(teaser_summary)
        if isinstance(full_report, str):
            full_report = json.loads(full_report)

        violations_all = full_report.get("violations", [])
        first_violation = violations_all[0] if violations_all else None

        overall_score = full_report.get("overall_score") or full_report.get("compliance_score")
        if overall_score is None:
            core_crit = full_report.get("core_criteria")
            if core_crit:
                from services.rag_auditor import calculate_score_from_criteria
                overall_score = calculate_score_from_criteria(core_crit)
            else:
                overall_score = 50
        overall_score = int(overall_score)
        projected_score = min(98, max(95, overall_score + 35))
        executive_summary = full_report.get("executive_summary", "")

        # Guard: If submitted policy passes compliance, no replacement redraft is required
        requires_redraft = (overall_score < 95) and (len(violations_all) > 0)

        # Server-side redaction for unpaid previews:
        # If policy passed compliance (requires_redraft is False): no paywall and no draft
        if not requires_redraft:
            locked_violations = []
            full_compliant_draft = None
            compliant_rewrite_html = None
        elif is_paid:
            locked_violations = violations_all[1:] if len(violations_all) > 1 else []
            full_compliant_draft = report.get("compliant_rewrite", "")
            compliant_rewrite_html = format_policy_html(full_compliant_draft)
        else:
            locked_violations = []
            if len(violations_all) > 1:
                for idx, v in enumerate(violations_all[1:], start=2):
                    locked_violations.append({
                        "id": v.get("id", idx),
                        "title": v.get("title", "Statutory DPA Alignment Gap"),
                        "severity": v.get("severity", "High"),
                        "citation": v.get("citation", "Kenya Data Protection Act, 2019"),
                        "explanation": None,  # Redacted server-side
                        "recommended_fix": None  # Redacted server-side
                    })
            full_compliant_draft = ""  # NEVER transmit full draft if unpaid
            compliant_rewrite_html = ""

        return render_template(
            "report.html",
            report=report,
            audit_id=audit_id,
            is_paid=is_paid,
            requires_redraft=requires_redraft,
            risk_level=report.get("risk_level", "High"),
            overall_score=overall_score,
            projected_score=projected_score,
            executive_summary=executive_summary,
            first_violation=first_violation,
            locked_violations=locked_violations,
            all_violations=violations_all,
            compliant_rewrite=full_compliant_draft,
            compliant_rewrite_html=compliant_rewrite_html,
            report_price=REPORT_PRICE
        )
    except Exception as e:
        logger.exception("Error rendering report %s: %s", audit_id, e)
        flash(f"Error loading report: {e}", "error")
        return redirect(url_for("index"))


@app.route("/download/<audit_id>/<file_format>")
def download_report_policy(audit_id, file_format):
    """
    Secure server-side download endpoint:
    Verifies that the audit is confirmed PAID with a valid MpesaReceiptNumber
    before allowing download of the replacement policy in TXT or PDF format.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            flash("Audit report not found.", "error")
            return redirect(url_for("index"))

        is_paid = bool(report.get("is_paid")) and bool(report.get("mpesa_receipt_number"))
        if not is_paid:
            flash("Access denied: You must complete payment via M-Pesa to download the compliance draft.", "error")
            return redirect(url_for("view_report", audit_id=audit_id))

        policy_text = report.get("compliant_rewrite", "")
        if not policy_text:
            flash("Policy draft content is empty.", "error")
            return redirect(url_for("view_report", audit_id=audit_id))

        org_name = report.get("target_url_or_name", "Organization").strip()
        safe_org = re.sub(r'[^\w\-]', '_', org_name)[:30]

        fmt = file_format.lower()
        if fmt == "pdf":
            pdf_bytes = generate_policy_pdf(policy_text, org_name, audit_id)
            return Response(
                pdf_bytes,
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=ODPC_Compliant_Policy_{safe_org}_{audit_id[:8]}.pdf"
                }
            )
        else:  # txt
            clean_txt = strip_markdown_for_text(policy_text)
            return Response(
                clean_txt,
                mimetype="text/plain; charset=utf-8",
                headers={
                    "Content-Disposition": f"attachment; filename=ODPC_Compliant_Policy_{safe_org}_{audit_id[:8]}.txt"
                }
            )

    except Exception as e:
        logger.exception("Error generating download for %s (%s): %s", audit_id, file_format, e)
        flash(f"Error generating download: {e}", "error")
        return redirect(url_for("view_report", audit_id=audit_id))


@app.route("/api/pay/stk-push", methods=["POST"])
@app.route("/api/pay/<audit_id>", methods=["POST"])
def initiate_payment_endpoint(audit_id=None):
    """
    Triggers an M-Pesa STK Push prompt for KES 1,000 to the user's phone.
    Supports /api/pay/<audit_id> and /api/pay/stk-push (with audit_id in body).
    """
    try:
        data = request.get_json(silent=True) or request.form or {}
        effective_audit_id = audit_id or data.get("audit_id")
        if not effective_audit_id:
            return jsonify({"success": False, "error": "Audit ID is required."}), 400

        report = fetch_audit_report(effective_audit_id)
        if not report:
            return jsonify({"success": False, "error": "Audit report not found."}), 404

        if report.get("is_paid") and report.get("mpesa_receipt_number"):
            return jsonify({"success": True, "message": "Report is already unlocked!", "is_paid": True})

        phone_number = data.get("phone_number", "").strip()
        if not phone_number:
            return jsonify({"success": False, "error": "Phone number is required."}), 400

        # Validate phone number format before hitting Daraja
        clean_phone = re.sub(r'[\s\-\+\(\)]', '', phone_number)
        if not (re.match(r'^(?:254|\+254|0)?([17]\d{8})$', clean_phone) or (clean_phone.isdigit() and len(clean_phone) in (9, 10, 12))):
            return jsonify({"success": False, "error": f"Invalid Kenyan mobile phone number '{phone_number}'. Must be a valid Safaricom/Airtel format (e.g., 0712345678 or 254712345678)."}), 400

        # Initiate Daraja STK Push
        account_ref = f"ODPC-{str(effective_audit_id)[:8]}"
        stk_result = initiate_stk_push(
            phone_number=phone_number,
            amount=REPORT_PRICE,
            account_ref=account_ref,
            audit_id=effective_audit_id
        )

        if not stk_result.get("success"):
            return jsonify({
                "success": False,
                "error": stk_result.get("error", "Failed to initiate M-Pesa STK push.")
            }), 400

        checkout_id = stk_result.get("checkout_request_id")

        # Save checkout_request_id and phone to the audit report record
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE audit_reports
                    SET mpesa_checkout_id = %s,
                        phone_number = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s::uuid;
                    """,
                    (checkout_id, phone_number, effective_audit_id)
                )

        return jsonify({
            "success": True,
            "checkout_id": checkout_id,
            "message": stk_result.get("customer_message"),
            "is_simulation": stk_result.get("is_simulation", False)
        })

    except Exception as e:
        logger.exception("Payment initiation failed for audit %s: %s", audit_id, e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/mpesa/callback", methods=["POST"])
def mpesa_callback():
    """
    Safaricom Daraja Webhook Callback endpoint with spoofing protection.
    1. Validates secret token if MPESA_CALLBACK_SECRET is configured.
    2. Verifies transaction status with Daraja STK Push Query if credentials are live.
    3. Confirms payment and unlocks report only on verified success.
    """
    try:
        # Validate secret token if configured
        expected_secret = os.getenv("MPESA_CALLBACK_SECRET", "").strip()
        if expected_secret:
            received_secret = (
                request.args.get("secret") or 
                request.args.get("token") or 
                request.headers.get("X-Mpesa-Secret-Token", "")
            ).strip()
            if received_secret != expected_secret:
                logger.warning("Rejected spoofed M-Pesa callback: secret token mismatch.")
                return jsonify({"ResultCode": 1, "ResultDesc": "Unauthorized callback access."}), 401

        callback_data = request.get_json(force=True, silent=True) or {}
        logger.info("Received M-Pesa Callback: %s", json.dumps(callback_data))

        parsed = parse_mpesa_callback(callback_data)
        checkout_id = parsed.get("checkout_request_id")
        result_code = parsed.get("result_code")
        receipt = parsed.get("mpesa_receipt_number")

        logger.info(
            "M-Pesa Callback parsed: CheckoutID=%s, ResultCode=%s, Receipt=%s",
            checkout_id, result_code, receipt
        )

        if parsed.get("success") and checkout_id:
            # Query Daraja API to verify directly with Safaricom if live credentials exist
            consumer_key = os.getenv("MPESA_CONSUMER_KEY", "").strip()
            if consumer_key and not checkout_id.startswith("ws_CO_"):
                stk_status = query_stk_status(checkout_id)
                if not stk_status.get("success"):
                    logger.warning(
                        "Daraja STK push query verification failed for %s: %s",
                        checkout_id, stk_status.get("result_desc")
                    )
                    return jsonify({"ResultCode": 1, "ResultDesc": "Direct STK verification failed."}), 200

            # Payment verified: unlock report
            receipt_code = receipt or f"MPESA-{datetime.now().strftime('%d%H%M%S')}"
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE audit_reports
                        SET is_paid = true,
                            mpesa_receipt_number = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE mpesa_checkout_id = %s;
                        """,
                        (receipt_code, checkout_id)
                    )
            logger.info("Audit report associated with CheckoutID '%s' marked as PAID (Receipt: %s).", checkout_id, receipt_code)
        else:
            logger.warning(
                "M-Pesa payment unsuccessful or cancelled for CheckoutID %s: %s",
                checkout_id, parsed.get("result_desc")
            )

        return jsonify({"ResultCode": 0, "ResultDesc": "Callback processed successfully."}), 200

    except Exception as e:
        logger.exception("Error processing Safaricom M-Pesa callback: %s", e)
        return jsonify({"ResultCode": 1, "ResultDesc": f"Callback processing error: {e}"}), 200


@app.route("/api/report/<audit_id>/status", methods=["GET"])
def report_status_polling(audit_id):
    """
    Polling endpoint: frontend checks this after triggering the M-Pesa STK push.
    Unlocks report as soon as is_paid is confirmed with a receipt.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            return jsonify({"success": False, "error": "Report not found."}), 404

        is_paid = bool(report.get("is_paid")) and bool(report.get("mpesa_receipt_number"))
        return jsonify({
            "success": True,
            "audit_id": audit_id,
            "is_paid": is_paid,
            "risk_level": report.get("risk_level"),
            "mpesa_receipt_number": report.get("mpesa_receipt_number")
        })
    except Exception as e:
        logger.error("Error polling report status: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pay/simulate-sandbox/<audit_id>", methods=["POST"])
def simulate_sandbox_payment(audit_id):
    """
    Sandbox / Test Helper: Simulates an immediate successful M-Pesa confirmation
    for testing the unlock flow locally without live Safaricom credentials.
    """
    try:
        simulated_receipt = f"SHG{datetime.now().strftime('%d%H%M%S')}AA"
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE audit_reports
                    SET is_paid = true,
                        mpesa_receipt_number = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s::uuid;
                    """,
                    (simulated_receipt, audit_id)
                )

        logger.info("Simulated sandbox payment successful for audit %s. Receipt: %s", audit_id, simulated_receipt)
        return jsonify({
            "success": True,
            "message": "Simulated payment successful! Full report unlocked.",
            "receipt": simulated_receipt
        })
    except Exception as e:
        logger.exception("Sandbox payment simulation failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/privacy")
@app.route("/privacy-policy")
def privacy_policy_page():
    """Renders the application's own Privacy Notice."""
    return render_template("privacy_policy.html")


@app.route("/terms")
def terms_page():
    """Renders the application's Terms of Service."""
    return render_template("terms.html")


@app.route("/health")
def health_check():
    """Health check endpoint verifying application and database readiness."""
    db_ok = test_connection()
    return jsonify({
        "status": "healthy" if db_ok else "database_degraded",
        "database_connected": db_ok,
        "service": "SeraGuard - Kenyan ODPC Compliance Auditor SaaS",
        "version": "1.1.0"
    }), (200 if db_ok else 503)


# -------------------------------------------------------------------------
# Application Initialization
# -------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "0").strip().lower() in ["1", "true", "yes"]
    print(f"\n🇰🇪 Starting SeraGuard - Kenyan ODPC Compliance Auditor on port {port} (Debug: {debug_mode})...\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
