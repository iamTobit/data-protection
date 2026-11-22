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
