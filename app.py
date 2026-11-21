"""
Kenyan Data Protection (ODPC) Compliance Auditor SaaS.
Flask Application Entry Point.
"""

import os
import json
import logging
from uuid import UUID
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from dotenv import load_dotenv

from database.connection import get_db_connection, test_connection, init_db
from services.rag_auditor import scrape_policy_from_url, run_compliance_audit
from services.mpesa import initiate_stk_push, parse_mpesa_callback, format_phone_number

# Load environment configuration
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger("odpc_auditor_app")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "odpc-compliance-secret-key-production-replace")
REPORT_PRICE = int(os.getenv("REPORT_UNLOCK_PRICE_KES", "1000"))


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
# Database Storage Helpers
# -------------------------------------------------------------------------
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
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, target_url_or_name, raw_input_text, risk_level,
                       teaser_summary, full_report, compliant_rewrite,
                       is_paid, mpesa_checkout_id, mpesa_receipt_number,
                       created_at, updated_at
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
    and the dual scanner input (URL scanner or raw text paste).
    """
    return render_template("index.html")


@app.route("/api/audit", methods=["POST"])
def run_audit_endpoint():
    """
    Endpoint that takes either a website URL OR raw text, executes the
    compliance audit RAG pipeline, saves the report, and returns the redirect URL.
    """
    try:
        # Support both form submission and JSON API request
        if request.is_json:
            data = request.get_json()
            scan_type = data.get("scan_type", "url")
            target_url = data.get("target_url", "").strip()
            policy_text = data.get("policy_text", "").strip()
        else:
            scan_type = request.form.get("scan_type", "url")
            target_url = request.form.get("target_url", "").strip()
            policy_text = request.form.get("policy_text", "").strip()

        document_text = ""
        source_label = ""

        if scan_type == "url":
            if not target_url:
                if request.is_json:
                    return jsonify({"success": False, "error": "Please provide a valid website URL."}), 400
                flash("Please enter a valid website URL.", "error")
                return redirect(url_for("index"))

            logger.info("Initiating scrape for: %s", target_url)
            scrape_res = scrape_policy_from_url(target_url)
            document_text = scrape_res["text"]
            source_label = scrape_res.get("title") or target_url
        else:
            if not policy_text or len(policy_text.strip()) < 100:
                msg = "Privacy Policy text must be at least 100 characters to perform a compliance audit."
                if request.is_json:
                    return jsonify({"success": False, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("index"))

            document_text = policy_text.strip()
            source_label = request.form.get("entity_name", "").strip() or "Submitted Policy Document"

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
    - If is_paid is FALSE: reveals overall risk level and ONLY the first violation.
      Remaining violations and the complete compliant rewrite are blurred behind
      the KES 1,000 M-Pesa unlock modal.
    - If is_paid is TRUE: displays the complete audit, all violations, and
      allows 1-click copying/downloading of the ODPC-compliant rewrite.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            flash("Audit report not found or expired.", "error")
            return redirect(url_for("index"))

        is_paid = report.get("is_paid", False)
        teaser_summary = report.get("teaser_summary") or {}
        full_report = report.get("full_report") or {}

        # Parse JSON fields if returned as string
        if isinstance(teaser_summary, str):
            teaser_summary = json.loads(teaser_summary)
        if isinstance(full_report, str):
            full_report = json.loads(full_report)

        violations_all = full_report.get("violations", [])
        first_violation = violations_all[0] if violations_all else None
        locked_violations = violations_all[1:] if len(violations_all) > 1 else []

        overall_score = full_report.get("overall_score", 45)
        executive_summary = full_report.get("executive_summary", "")

        return render_template(
            "report.html",
            report=report,
            audit_id=audit_id,
            is_paid=is_paid,
            risk_level=report.get("risk_level", "High"),
            overall_score=overall_score,
            executive_summary=executive_summary,
            first_violation=first_violation,
            locked_violations=locked_violations,
            all_violations=violations_all,
            compliant_rewrite=report.get("compliant_rewrite", ""),
            report_price=REPORT_PRICE
        )
    except Exception as e:
        logger.exception("Error rendering report %s: %s", audit_id, e)
        flash(f"Error loading report: {e}", "error")
        return redirect(url_for("index"))


@app.route("/api/pay/<audit_id>", methods=["POST"])
def initiate_payment_endpoint(audit_id):
    """
    Triggers an M-Pesa STK Push prompt for KES 1,000 to the user's phone.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            return jsonify({"success": False, "error": "Audit report not found."}), 404

        if report.get("is_paid"):
            return jsonify({"success": True, "message": "Report is already unlocked!", "is_paid": True})

        data = request.get_json() or request.form
        phone_number = data.get("phone_number", "").strip()

        if not phone_number:
            return jsonify({"success": False, "error": "Phone number is required."}), 400

        # Initiate Daraja STK Push
        account_ref = f"ODPC-{audit_id[:8]}"
        stk_result = initiate_stk_push(
            phone_number=phone_number,
            amount=REPORT_PRICE,
            account_ref=account_ref,
            audit_id=audit_id
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
                    (checkout_id, phone_number, audit_id)
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
    Safaricom Daraja Webhook Callback endpoint.
    Verifies ResultCode == 0, marks is_paid=True in PostgreSQL, and logs transaction details.
    """
    try:
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
            # Payment confirmed: unlock report
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
                        (receipt or "DAR-MPESA-PAID", checkout_id)
                    )
            logger.info("Audit report associated with CheckoutID '%s' marked as PAID.", checkout_id)
        else:
            logger.warning(
                "M-Pesa payment unsuccessful or cancelled for CheckoutID %s: %s",
                checkout_id, parsed.get("result_desc")
            )

        # Safaricom expects standard acknowledgement JSON
        return jsonify({"ResultCode": 0, "ResultDesc": "Callback processed successfully."}), 200

    except Exception as e:
        logger.exception("Error processing Safaricom M-Pesa callback: %s", e)
        return jsonify({"ResultCode": 1, "ResultDesc": f"Callback processing error: {e}"}), 200


@app.route("/api/report/<audit_id>/status", methods=["GET"])
def report_status_polling(audit_id):
    """
    Polling endpoint: frontend checks this every 2 seconds after triggering
    the M-Pesa STK push. The instant the callback sets is_paid=True, the frontend
    refreshes and unlocks the full report without manual reload.
    """
    try:
        report = fetch_audit_report(audit_id)
        if not report:
            return jsonify({"success": False, "error": "Report not found."}), 404

        return jsonify({
            "success": True,
            "audit_id": audit_id,
            "is_paid": bool(report.get("is_paid", False)),
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
    for testing the unlock flow locally without waiting for an actual Safaricom phone push.
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
            "message": "Simulated payment successful! The full report is now unlocked.",
            "receipt": simulated_receipt
        })
    except Exception as e:
        logger.exception("Sandbox payment simulation failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint verifying application and database readiness."""
    db_ok = test_connection()
    return jsonify({
        "status": "healthy" if db_ok else "database_degraded",
        "database_connected": db_ok,
        "service": "Kenyan ODPC Compliance Auditor SaaS",
        "version": "1.0.0"
    }), (200 if db_ok else 503)


# -------------------------------------------------------------------------
# Application Initialization
# -------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in ["1", "true", "yes"]
    print(f"\n🇰🇪 Starting Kenyan ODPC Compliance Auditor on port {port} (Debug: {debug_mode})...\n")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
