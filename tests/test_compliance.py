"""
Comprehensive Test Suite for Kenyan ODPC Compliance Auditor SaaS.
Tests:
1. Phone number formatting for Safaricom Daraja M-Pesa.
2. Daraja callback parsing and validation.
3. HTML cleaning and text extraction.
4. Deterministic 384-dimensional embedding normalization.
5. Statutory compliance evaluation engine and teaser generation.
6. Flask application route handling.
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.mpesa import format_phone_number, generate_stk_password, parse_mpesa_callback
from services.rag_auditor import (
    clean_html_text, 
    _deterministic_384d_embedding, 
    generate_embedding,
    _audit_with_heuristics, 
    create_teaser_summary
)
import app as flask_app_module


class TestMpesaService(unittest.TestCase):
    """Test Safaricom Daraja Lipa Na M-Pesa functionality."""

    def test_phone_formatting(self):
        """Tests that Kenyan phone formats are normalized to 254XXXXXXXXX."""
        self.assertEqual(format_phone_number("0712345678"), "254712345678")
        self.assertEqual(format_phone_number("0112345678"), "254112345678")
        self.assertEqual(format_phone_number("+254712345678"), "254712345678")
        self.assertEqual(format_phone_number("254712345678"), "254712345678")
        self.assertEqual(format_phone_number("712345678"), "254712345678")
        self.assertEqual(format_phone_number("0712-345-678"), "254712345678")

        with self.assertRaises(ValueError):
            format_phone_number("12345")

        with self.assertRaises(ValueError):
            format_phone_number("0201234567")  # Landline

    def test_stk_password_generation(self):
        """Verifies base64 STK push password generation."""
        pwd = generate_stk_password("174379", "bfb279f9aa9bdbcf", "20261121170000")
        self.assertTrue(isinstance(pwd, str))
        self.assertTrue(len(pwd) > 20)

    def test_parse_mpesa_callback_success(self):
        """Verifies successful Safaricom callback parsing."""
        callback_payload = {
            "Body": {
                "stkCallback": {
                    "MerchantRequestID": "29115-34620561-1",
                    "CheckoutRequestID": "ws_CO_19122026102115052",
                    "ResultCode": 0,
                    "ResultDesc": "The service request is processed successfully.",
                    "CallbackMetadata": {
                        "Item": [
                            {"Name": "Amount", "Value": 1000.00},
                            {"Name": "MpesaReceiptNumber", "Value": "SHG78291AA"},
                            {"Name": "TransactionDate", "Value": 20261121171000},
                            {"Name": "PhoneNumber", "Value": 254712345678}
                        ]
                    }
                }
            }
        }
        res = parse_mpesa_callback(callback_payload)
        self.assertTrue(res["success"])
        self.assertEqual(res["result_code"], 0)
        self.assertEqual(res["checkout_request_id"], "ws_CO_19122026102115052")
        self.assertEqual(res["mpesa_receipt_number"], "SHG78291AA")
        self.assertEqual(res["amount"], 1000.0)
        self.assertEqual(res["phone_number"], "254712345678")

    def test_parse_mpesa_callback_failure(self):
        """Verifies cancelled or failed M-Pesa transaction parsing."""
        failed_payload = {
            "Body": {
                "stkCallback": {
                    "MerchantRequestID": "29115-34620561-1",
                    "CheckoutRequestID": "ws_CO_FAILED_123",
                    "ResultCode": 1032,
                    "ResultDesc": "Request cancelled by user."
                }
            }
        }
        res = parse_mpesa_callback(failed_payload)
        self.assertFalse(res["success"])
        self.assertEqual(res["result_code"], 1032)
        self.assertEqual(res["checkout_request_id"], "ws_CO_FAILED_123")
        self.assertIsNone(res["mpesa_receipt_number"])


class TestRAGAuditor(unittest.TestCase):
    """Test text cleaning, 384d embedding generation, and legal heuristics."""

    def test_clean_html_text(self):
        """Tests that scripts, styles, and tags are cleaned."""
        html = """
        <html>
            <head><script>alert('bad');</script><style>.bad{color:red;}</style></head>
            <body>
                <header><nav><a href='/'>Home</a></nav></header>
                <h1>Nairobi E-Commerce Privacy Notice</h1>
                <p>We collect your personal information to fulfill orders and send marketing SMS.</p>
                <footer><p>Copyright 2026</p></footer>
            </body>
        </html>
        """
        cleaned = clean_html_text(html)
        self.assertNotIn("alert('bad')", cleaned)
        self.assertNotIn(".bad", cleaned)
        self.assertIn("Nairobi E-Commerce Privacy Notice", cleaned)
        self.assertIn("marketing SMS", cleaned)

    def test_deterministic_384d_embedding(self):
        """Verifies 384-dimensional normalized vector generation."""
        text = "Conditions of consent under Section 32 of Kenya Data Protection Act 2019"
        vec = _deterministic_384d_embedding(text)
        self.assertEqual(len(vec), 384)
        
        # Check L2 normalization: sum(x^2) should be approximately 1.0
        norm_sq = sum(x * x for x in vec)
        self.assertAlmostEqual(norm_sq, 1.0, places=4)

    def test_heuristic_audit_and_teaser(self):
        """Tests statutory compliance evaluation on a non-compliant policy."""
        non_compliant_text = """
        Privacy Policy of QuickLoans Kenya.
        We access your contacts and send you daily promotional SMS without opt-out.
        We store your data indefinitely for business archives.
        All data is hosted in foreign cloud servers with no specific safeguards.
        """
        report = _audit_with_heuristics(non_compliant_text, [], "QuickLoans Kenya")
        
        self.assertEqual(report["risk_level"], "High")
        self.assertTrue(len(report["violations"]) >= 3)
        self.assertTrue(any("37" in v["citation"] for v in report["violations"]))
        self.assertTrue(any("39" in v["citation"] for v in report["violations"]))
        self.assertIn("complaints@odpc.go.ke", report["compliant_policy_draft"])

        # Test teaser generation
        teaser = create_teaser_summary(report)
        self.assertEqual(teaser["risk_level"], "High")
        self.assertIsNotNone(teaser["unlocked_first_violation"])
        self.assertTrue(teaser["locked_count"] >= 2)


class TestSectorProfilingAndDynamicRedraft(unittest.TestCase):
    """Test context-aware sector profiling and tailored non-bloated policy redrafting."""

    def test_education_sector_profiling_and_redraft(self):
        """Verifies high school detection and zero e-commerce/fintech junk in redraft."""
        from services.rag_auditor import detect_operational_sector, generate_static_policy_draft, _audit_with_heuristics

        text = (
            "Alliance Girls High School Privacy Notice. P.O. Box 123 Nairobi, Kenya. "
            "We collect student academic progress, learner admission records, KCSE registration, "
            "parent emergency phone numbers (0712345678), and school fee receipts via M-Pesa. "
            "Student personal data is processed pursuant to the Basic Education Act. "
            "Parental consent is obtained at admission for all minors under 18 pursuant to Section 33. "
            "Medical emergency records are kept by the school nurse under Section 44. "
            "Retention: academic certificates permanent, student files 7 years. "
            "72-hour breach notification to ODPC under Section 43. "
            "ODPC complaints can be sent to complaints@odpc.go.ke at Britam Tower, Nairobi. "
            "We uphold principles of data minimisation and accuracy under Section 25. "
            "Data subject rights of access, correction, erasure, portability, objection honored within 30 days."
        )
        sector_info = detect_operational_sector("Alliance Girls High School", "alliancegirls.sc.ke", text)
        self.assertEqual(sector_info["sector"], "education")
        self.assertTrue(sector_info["has_minors"])

        # Verify heuristic audit marks non-commercial direct marketing as N/A
        audit_res = _audit_with_heuristics(text, [], "Alliance Girls High School", "alliancegirls.sc.ke", sector_info)
        marketing_crit = [c for c in audit_res["core_criteria"] if c["key"] == "direct_marketing_opt_out"][0]
        self.assertEqual(marketing_crit["status"], "N/A")
        self.assertEqual(marketing_crit["score"], 10)
        self.assertGreaterEqual(audit_res["overall_score"], 90)

        # Verify static draft has education focus and zero retail/cart junk
        draft = generate_static_policy_draft("Alliance Girls High School", "alliancegirls.sc.ke", "privacy@alliancegirls.sc.ke", sector_info)
        self.assertIn("EDUCATION SECTOR", draft)
        self.assertIn("Learner & Student Academic Data", draft)
        self.assertNotIn("shopping cart", draft.lower())
        self.assertNotIn("dispatch rider", draft.lower())
        self.assertNotIn("credit scoring", draft.lower())

    def test_healthcare_sector_profiling_and_redraft(self):
        """Verifies clinic/hospital detection and Section 44 medical confidentiality in redraft."""
        from services.rag_auditor import detect_operational_sector, generate_static_policy_draft

        text = (
            "AAR Health Clinic Nairobi. We collect patient demographics, next-of-kin, medical history, "
            "clinical consultation records, laboratory diagnostic results, prescriptions, and SHA / insurance "
            "billing details under Section 44 of the Kenya Data Protection Act 2019 for healthcare care. "
            "Confidential patient records are retained for 10 years pursuant to Ministry of Health guidelines. "
            "We do not sell patient data for commercial marketing. Section 43 breach notifications are made within 72 hours. "
            "Contact ODPC at Britam Tower, Nairobi."
        )
        sector_info = detect_operational_sector("AAR Health Clinic Nairobi", "aarhealthcare.com", text)
        self.assertEqual(sector_info["sector"], "healthcare")
        self.assertTrue(sector_info["has_health_data"])

        draft = generate_static_policy_draft("AAR Health Clinic Nairobi", "aarhealthcare.com", "privacy@aarhealthcare.com", sector_info)
        self.assertIn("HEALTHCARE SECTOR", draft)
        self.assertIn("Sensitive Health Data", draft)
        self.assertIn("Section 44(1)(f)", draft)
        self.assertNotIn("shopping cart", draft.lower())
        self.assertNotIn("courier", draft.lower())
        self.assertNotIn("credit appraisal", draft.lower())

    def test_retail_sector_profiling_and_redraft(self):
        """Verifies retail/e-commerce detection and order/courier focus in redraft."""
        from services.rag_auditor import detect_operational_sector, generate_static_policy_draft

        text = (
            "PrimeTech Kenya Online Electronics Store. We collect customer names, delivery addresses, "
            "dispatch rider tracking, customer orders, M-Pesa payment confirmation codes, and shopping cart cookies. "
            "We seek prior consent for promotional newsletters with an unburdened 48-hour opt-out mechanism. "
            "Financial sales records are kept for 7 years for KRA tax purposes. We notify ODPC within 72 hours of security breaches. "
            "ODPC recourse at Britam Tower."
        )
        sector_info = detect_operational_sector("PrimeTech Kenya Online Store", "primetech.ke", text)
        self.assertEqual(sector_info["sector"], "retail")
        self.assertTrue(sector_info["has_ecommerce"])

        draft = generate_static_policy_draft("PrimeTech Kenya Online Store", "primetech.ke", "privacy@primetech.ke", sector_info)
        self.assertIn("RETAIL & E-COMMERCE", draft)
        self.assertIn("Customer Account & Contact Details", draft)
        self.assertNotIn("patient", draft.lower())
        self.assertNotIn("learner", draft.lower())


class TestFlaskRoutes(unittest.TestCase):
    """Test web application routes and endpoints."""

    def setUp(self):
        flask_app_module.app.config["TESTING"] = True
        flask_app_module.app.config["SECRET_KEY"] = "test-secret-key"
        self.client = flask_app_module.app.test_client()

    def test_landing_page(self):
        """Verifies GET / returns 200 with ODPC branding and form tabs."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("SeraGuard", content)
        self.assertIn("Kenya Data Protection Act, 2019", content)
        self.assertIn("Scan URL", content)
        self.assertIn("Upload File", content)
        self.assertIn("Paste Text", content)

    def test_health_endpoint(self):
        """Verifies GET /health returns JSON."""
        response = self.client.get("/health")
        # May be 200 or 503 depending on database connection
        self.assertIn(response.status_code, [200, 503])
        data = response.get_json()
        self.assertEqual(data["service"], "SeraGuard - Kenyan ODPC Compliance Auditor SaaS")


if __name__ == "__main__":
    unittest.main()
