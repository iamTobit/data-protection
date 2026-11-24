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
    create_teaser_summary,
    extract_clean_entity_and_domain,
    get_candidate_llm_targets
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

    def test_extract_clean_entity_and_domain_filename_noise(self):
        """Verifies filenames with Appendix and codes don't become fake domains."""
        filename = "Kabu 13010.Appendix I Kabarak University Data Privacy Statement (1).pdf"
        entity, domain = extract_clean_entity_and_domain(filename, "Kabarak University collects student records. Contact: info@kabarak.ac.ke")
        self.assertEqual(entity, "Kabarak University")
        self.assertEqual(domain, "kabarak.ac.ke")
        self.assertNotIn("13010", domain)
        self.assertNotIn("appendix", domain)

    def test_extract_clean_entity_and_domain_naked_url(self):
        """Verifies naked domain or clean URL extraction."""
        entity, domain = extract_clean_entity_and_domain("https://aarhospital.com/privacy-policy")
        self.assertEqual(domain, "aarhospital.com")
        self.assertIn("Aarhospital", entity)

    def test_section_63_exact_wording_and_conditional_dpo(self):
        """Verifies Section 63 contains 'whichever is lower' and DPO is conditional under Section 24."""
        non_compliant_text = """
        Privacy Policy of BetaShop Kenya.
        We access your contacts and send you daily promotional SMS without opt-out.
        """
        report = _audit_with_heuristics(non_compliant_text, [], "BetaShop Kenya")
        # Section 63 exact statutory wording test
        self.assertIn("whichever is lower", report["executive_summary"])
        self.assertIn("5,000,000", report["executive_summary"])
        self.assertNotIn("whichever is higher", report["executive_summary"])

        # Conditional DPO test
        rights_violation = next((v for v in report["violations"] if "Incomplete Enumeration" in v.get("title", "")), None)
        self.assertIsNotNone(rights_violation)
        self.assertIn("where mandatory under Section 24", rights_violation["recommended_fix"])

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "gemini")
    def test_multi_provider_gemini_priority(self):
        """Verifies Gemini is prioritized when LLM_PROVIDER=gemini."""
        import services.rag_auditor as rag
        targets = rag.get_candidate_llm_targets()
        self.assertTrue(len(targets) > 0)
        self.assertEqual(targets[0]["provider"], "gemini")
        self.assertEqual(targets[0]["model"], "gemini-3.5-flash")
        self.assertEqual(targets[0]["api_key"], "test-gemini-key")
        self.assertIn("googleapis.com", targets[0]["base_url"])

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "openai")
    @patch("services.rag_auditor.LLM_API_KEY", "test-groq-key")
    def test_multi_provider_gemini_fallback(self):
        """Verifies Gemini is included as fallback with trailing slash."""
        import services.rag_auditor as rag
        targets = rag.get_candidate_llm_targets()
        models = [t["model"] for t in targets]
        self.assertIn("gemini-3.5-flash", models)
        self.assertIn("gemini-3.8-flash", models)
        self.assertIn("gemini-3.5-flash-lite", models)
        gemini_targets = [t for t in targets if t["provider"] == "gemini"]
        self.assertTrue(len(gemini_targets) >= 1)
        for gt in gemini_targets:
            self.assertTrue(gt["base_url"].endswith("/"), f"Gemini base_url must end with trailing slash: {gt['base_url']}")
            self.assertEqual(gt["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai/")

    def test_unedited_template_placeholders_detection(self):
        """Verifies raw [Insert ...] placeholders downgrade Criterion 1 to FAIL (0 pts) and add High gap."""
        policy_with_placeholders = """
        Privacy Notice of EMA Advocates. Located in Nairobi, Kenya.
        Contact us at [Insert Email Address] or phone [Insert Phone Number].
        We process client legal data under advocate service contracts. We do not transfer overseas.
        """
        report = _audit_with_heuristics(policy_with_placeholders, [], "EMA Advocates")
        crit_1 = next((c for c in report["core_criteria"] if c["key"] == "data_controller_identity"), None)
        self.assertIsNotNone(crit_1)
        self.assertEqual(crit_1["status"], "FAIL")
        self.assertEqual(crit_1["score"], 0)
        placeholder_violation = next((v for v in report["violations"] if "Placeholder" in v["title"]), None)
        self.assertIsNotNone(placeholder_violation)
        self.assertEqual(placeholder_violation["severity"], "High")
        self.assertTrue(report["overall_score"] <= 65)

    def test_determine_risk_level_standardization(self):
        """Verifies deterministic risk_level calculation overrides loose labels based on score and severity."""
        from services.rag_auditor import determine_risk_level

        # >= 85 and zero Critical/High gaps -> Low
        self.assertEqual(determine_risk_level(85, []), "Low")
        self.assertEqual(determine_risk_level(95, [{"severity": "Medium"}, {"severity": "Low"}]), "Low")

        # >= 85 with a High gap -> Medium
        self.assertEqual(determine_risk_level(90, [{"severity": "High"}]), "Medium")
        self.assertEqual(determine_risk_level(95, [{"severity": "Low"}, {"severity": "High"}]), "Medium")

        # >= 85 with a Critical gap -> Medium (capped at Medium)
        self.assertEqual(determine_risk_level(95, [{"severity": "Critical"}]), "Medium")

        # 55 <= score <= 84 -> Strictly Medium (regardless of Critical or High gaps)
        self.assertEqual(determine_risk_level(55, []), "Medium")
        self.assertEqual(determine_risk_level(84, []), "Medium")
        self.assertEqual(determine_risk_level(70, [{"severity": "Medium"}]), "Medium")
        self.assertEqual(determine_risk_level(70, [{"severity": "High"}]), "Medium")
        self.assertEqual(determine_risk_level(67, [{"severity": "Critical"}]), "Medium")
        self.assertEqual(determine_risk_level(70, [{"severity": "Critical"}]), "Medium")

        # < 55 -> High (always)
        self.assertEqual(determine_risk_level(54, []), "High")
        self.assertEqual(determine_risk_level(30, [{"severity": "Low"}]), "High")
        self.assertEqual(determine_risk_level(0, []), "High")

    def test_save_audit_report_uses_clean_entity_over_noisy_filename(self):
        """Verifies save_audit_report saves clean entity name instead of raw noisy filename."""
        import app as flask_app
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"id": "mock-uuid-1234"}
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_conn
        mock_cm.__exit__.return_value = False

        with patch("app.get_db_connection", return_value=mock_cm):
            # Test when target is a raw PDF filename with Appendix noise
            raw_filename = "Kabu 13010.Appendix I Kabarak University Data Privacy Statement (1).pdf"
            audit_result = {
                "risk_level": "Medium",
                "compliant_rewrite": "# Compliant Draft",
                "full_report": {
                    "entity_name": "Kabarak University",
                    "domain": "kabarak.ac.ke",
                    "overall_score": 75,
                    "violations": []
                },
                "teaser_summary": {
                    "entity_name": "Kabarak University"
                }
            }
            flask_app.save_audit_report(raw_filename, "Sample policy text for Kabarak", audit_result)
            
            # Check what target_url_or_name was passed to INSERT
            self.assertTrue(mock_cursor.execute.called)
            execute_args = mock_cursor.execute.call_args[0]
            inserted_params = execute_args[1]
            saved_target_name = inserted_params[0]
            self.assertEqual(saved_target_name, "Kabarak University")

    def test_auto_convert_unrepresented_partial_fail_criteria_to_gaps(self):
        """Verifies unrepresented FAIL/PARTIAL criteria are converted to Medium gap cards when gaps < 2 and score < 90."""
        from services.rag_auditor import run_compliance_audit

        mock_llm_report = {
            "overall_score": 75,
            "core_criteria": [
                {
                    "key": "data_controller_identity",
                    "name": "Data Controller Identity & Physical Address (Sec 24, 29)",
                    "status": "PASS",
                    "score": 10,
                    "finding": "Identity provided."
                },
                {
                    "key": "retention_schedule",
                    "name": "Concrete Data Retention Schedule & Disposal (Sec 39, Reg 19)",
                    "status": "PARTIAL",
                    "score": 5,
                    "finding": "General retention language used without concrete statutory periods."
                },
                {
                    "key": "children_data_protection",
                    "name": "Protection of Minors & Age 18 Verification (Sec 33)",
                    "status": "PARTIAL",
                    "score": 5,
                    "finding": "Missing age verification safeguards."
                }
            ],
            "violations": []
        }

        sample_doc = "This is a valid privacy policy document for Test Org that meets the minimum length requirement for compliance auditing under Kenyan DPA."
        with patch("services.rag_auditor._audit_with_llm", return_value=mock_llm_report), \
             patch("services.rag_auditor.get_candidate_llm_targets", return_value=[{"model": "test"}]):
            res = run_compliance_audit(sample_doc, "Test Org")
            full_rep = res["full_report"]
            gaps = full_rep.get("violations", [])
            self.assertGreaterEqual(len(gaps), 2)
            self.assertTrue(any("Retention" in g["title"] for g in gaps))
            self.assertTrue(any("Minors" in g["title"] or "Children" in g["title"] for g in gaps))
            for g in gaps:
                self.assertEqual(g["severity"], "Medium")

    def test_audit_cache_version_is_v12(self):
        """Verifies AUDIT_CACHE_VERSION is bumped to v12 in app.py."""
        import app as flask_app
        self.assertEqual(flask_app.AUDIT_CACHE_VERSION, "v12")

    def test_static_policy_drafts_contain_no_pipe_tables(self):
        """Verifies that static fallback redrafts across all sectors use bolded bullet lists instead of pipe tables."""
        from services.rag_auditor import generate_static_policy_draft
        sectors = [
            ("Kabarak University", "kabarak.ac.ke"),
            ("Avenue Hospital", "avenuehealthcare.com"),
            ("Zenka Digital", "zenka.co.ke"),
            ("Jumia Kenya", "jumia.co.ke"),
            ("Corporate Consulting", "corpconsult.co.ke")
        ]
        for name, domain in sectors:
            draft = generate_static_policy_draft(name, domain)
            self.assertNotIn("| :---", draft, f"Found pipe table divider in {name} draft")
            self.assertNotIn("|---", draft, f"Found pipe table divider in {name} draft")
            # Verify bolded bullet list exists in Section 4
            self.assertIn("### 4. Purposes and Lawful Bases", draft, f"Section 4 missing in {name} draft")
            self.assertIn("- **", draft, f"Expected bolded bullets in {name} draft")

    def test_strip_markdown_for_text_converts_legacy_tables(self):
        """Verifies strip_markdown_for_text converts legacy pipe tables to clean bullet lines."""
        from services.rag_auditor import strip_markdown_for_text
        legacy_markdown = (
            "### 4. Purposes and Lawful Bases\n"
            "| Processing Activity | Lawful Basis |\n"
            "| :--- | :--- |\n"
            "| Order fulfillment | **Contractual Performance** |\n"
            "| Tax auditing | **Legal Obligation** |\n"
        )
        plain_text = strip_markdown_for_text(legacy_markdown)
        self.assertNotIn("|", plain_text)
        self.assertNotIn(":---", plain_text)
        self.assertIn("• Processing Activity: Lawful Basis", plain_text)
        self.assertIn("• Order fulfillment: Contractual Performance", plain_text)
        self.assertIn("• Tax auditing: Legal Obligation", plain_text)

    def test_generate_policy_pdf_with_legacy_table_and_markdown(self):
        """Verifies generate_policy_pdf processes legacy pipe tables and bold markdown without crashing."""
        from services.rag_auditor import generate_policy_pdf
        legacy_policy = (
            "# ODPC-COMPLIANT PRIVACY POLICY\n"
            "**Governed under the Kenya Data Protection Act, 2019**\n\n"
            "### 1. Introduction\n"
            "We are committed to protecting personal data.\n\n"
            "### 4. Purposes & Lawful Bases\n"
            "| Activity | Lawful Basis |\n"
            "| :--- | :--- |\n"
            "| Student admission | **Contractual Mandate** |\n"
            "| Exam registration | **Legal Obligation** |\n\n"
            "- **M-Pesa & Tax Records:** Retained for **7 years**.\n\n"
            "*Legal Disclaimer: AI-assisted readiness instrument.*"
        )
        pdf_bytes = generate_policy_pdf(legacy_policy, "Kabarak University", "test-audit-12345678")
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_html_table_preservation_in_clean_html_text(self):
        """Verifies HTML table rows with short cells are preserved as pipe-separated text."""
        from services.rag_auditor import clean_html_text
        html = """
        <html><body>
            <h2>6. Data Retention Schedules</h2>
            <table>
                <tr><th>Category</th><th>Period</th><th>Rationale</th></tr>
                <tr><td>Tax records</td><td>7 years</td><td>KRA obligation</td></tr>
                <tr><td>Audit logs</td><td>90 days</td><td>Download access</td></tr>
                <tr><td>Security logs</td><td>30 days</td><td>Abuse mitigation</td></tr>
                <tr><td>Analytics</td><td>Indefinitely</td><td>Anonymised macro trends</td></tr>
            </table>
        </body></html>
        """
        cleaned = clean_html_text(html)
        self.assertIn("7 years", cleaned)
        self.assertIn("90 days", cleaned)
        self.assertIn("30 days", cleaned)
        self.assertIn("Tax records | 7 years | KRA obligation", cleaned)

    def test_ignore_indefinitely_when_data_anonymised(self):
        """Verifies indefinitely is ignored when data is explicitly anonymised."""
        from services.rag_auditor import _audit_with_heuristics
        policy_text = """
        Privacy Notice of DataCorp Ltd.
        Contact: privacy@datacorp.co.ke, Nairobi, Kenya.
        We process customer account data. Retention periods:
        Tax records are kept for 7 years. Security logs are retained for 30 days.
        Aggregated, anonymised analytics are kept indefinitely as they contain no personal data.
        You may opt-out of marketing free of charge at any time. ODPC recourse at Britam Tower.
        """
        report = _audit_with_heuristics(policy_text, [], "DataCorp Ltd")
        retention_crit = next(c for c in report["core_criteria"] if c["key"] == "retention_schedule")
        self.assertEqual(retention_crit["status"], "PASS")
        self.assertEqual(retention_crit["score"], 10)
        retention_violation = next((v for v in report["violations"] if "retention" in v["title"].lower()), None)
        self.assertIsNone(retention_violation)

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "openai")
    @patch("services.rag_auditor.LLM_API_KEY", "test-groq-key")
    def test_smart_length_based_provider_routing(self):
        """Verifies documents > 9000 chars route to Gemini first, while shorter docs route to Groq first."""
        import services.rag_auditor as rag
        # Normal length doc (<= 9000 chars) -> Groq is first
        short_targets = rag.get_candidate_llm_targets(text_length=5000)
        self.assertEqual(short_targets[0]["provider"], "groq/openai")

        # Long multi-page doc (> 9000 chars) -> Gemini is prioritized FIRST due to 1M token context
        long_targets = rag.get_candidate_llm_targets(text_length=15000)
        self.assertEqual(long_targets[0]["provider"], "gemini")
        self.assertEqual(long_targets[0]["model"], "gemini-3.5-flash")

    def test_clean_html_text_with_trafilatura(self):
        """Verifies clean_html_text strips nav/footer bloat and appends JSON-LD metadata."""
        from services.rag_auditor import clean_html_text
        sample_html = """
        <html>
        <head>
          <script type="application/ld+json">
          {"name": "Kenya Tech Ltd", "telephone": "+254700111222", "address": "Westlands, Nairobi"}
          </script>
        </head>
        <body>
          <nav><a href="/">Home</a> <a href="/cart">Cart (3)</a> <a href="/shop">Products</a></nav>
          <main>
            <h1>Privacy Notice</h1>
            <p>Kenya Tech Ltd processes customer personal data under the Kenya Data Protection Act 2019.</p>
            <p>We collect names, emails, and phone numbers to fulfill orders and communicate with users.</p>
            <p>Data subjects have rights under Section 26 to access, correct, or delete personal data.</p>
            <p>Complaints can be lodged with the ODPC at Britam Tower, Nairobi under Section 56.</p>
            <p>We implement strict encryption and security safeguards pursuant to Section 43.</p>
            <p>Data retention is limited to 7 years in compliance with statutory financial and tax obligations.</p>
          </main>
          <footer><p>Copyright 2026. All rights reserved.</p></footer>
        </body>
        </html>
        """
        extracted = clean_html_text(sample_html)
        self.assertNotIn("Cart (3)", extracted)
        self.assertIn("Kenya Tech Ltd", extracted)
        self.assertIn("Westlands, Nairobi", extracted)
        self.assertIn("Section 26", extracted)

    def test_bot_wall_guard_and_insufficient_content(self):
        """Verifies check_bot_wall_or_insufficient_content flags short text and anti-bot walls."""
        from services.rag_auditor import check_bot_wall_or_insufficient_content
        # Short text (< 350 chars)
        self.assertIsNotNone(check_bot_wall_or_insufficient_content("Short policy text."))
        # Bot wall challenges
        radware_html = "<html><body><h1>Verifying your browser</h1><p>Radware Bot Protection</p></body></html>"
        self.assertIsNotNone(check_bot_wall_or_insufficient_content("A" * 400, radware_html))
        cloudflare_text = "Enable JavaScript and cookies to continue. " + ("A" * 400)
        self.assertIsNotNone(check_bot_wall_or_insufficient_content(cloudflare_text))
        # Valid text
        valid_text = "Valid Privacy Policy text under Kenya Data Protection Act 2019. " * 15
        self.assertIsNone(check_bot_wall_or_insufficient_content(valid_text, "<html><body>...</body></html>"))

    @patch("services.rag_auditor.GEMINI_API_KEY", "")
    @patch("services.rag_auditor.LLM_API_KEY", "fake-key")
    @patch("openai.OpenAI")
    def test_audit_with_llm_chunk_and_token_capping(self, mock_openai_cls):
        """Verifies _audit_with_llm caps at top 8 legal chunks, uses Head+Tail smart sampling on Groq, max_tokens=3400, and reasoning_effort=medium."""
        from services.rag_auditor import _audit_with_llm
        from unittest.mock import MagicMock

        # Create mock client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"overall_score": 85, "core_criteria": [], "violations": []}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        # Create 17 fake chunks with distances
        chunks = [
            {"id": i, "act_title": "DPA 2019", "section_ref": f"Sec {i}", "citation": f"Citation {i}", "content": f"Content {i}", "distance": 0.05 * i}
            for i in range(1, 18)
        ]
        long_text = "HEAD_" + ("A" * 6000) + "_MIDDLE_" + ("Z" * 5000) + "_TAIL"  # > 11,000 chars

        _audit_with_llm(long_text, chunks, "Test Corp", "test.co.ke")

        self.assertTrue(mock_client.chat.completions.create.called)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 3400)
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(call_kwargs.get("reasoning_effort"), "medium")

        # Inspect the user prompt passed to the model
        messages = call_kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        
        # Verify only 8 chunks were included
        citation_count = user_msg.count("--- Citation")
        self.assertEqual(citation_count, 8)

        # Verify Groq uses Head + Tail sampling preserving both HEAD and TAIL
        policy_section = user_msg.split("<policy_text>")[-1].split("</policy_text>")[0].strip()
        self.assertIn("HEAD_", policy_section)
        self.assertIn("_TAIL", policy_section)
        self.assertIn("MIDDLE SECTIONS CONDENSED", policy_section)

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "openai")
    @patch("services.rag_auditor.LLM_API_KEY", "fake-key")
    @patch("openai.OpenAI")
    def test_audit_with_llm_gemini_full_text(self, mock_openai_cls):
        """Verifies Gemini receives full text without slicing on long documents."""
        from services.rag_auditor import _audit_with_llm
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"overall_score": 95, "core_criteria": [], "violations": []}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        # 15,000 char document
        full_text = ("Kabarak University Privacy Policy. " * 450).strip()
        self.assertTrue(len(full_text) > 9000)

        _audit_with_llm(full_text, [], "Kabarak University", "kabarak.ac.ke")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # Should call Gemini first with 8192 max_tokens and JSON mode
        self.assertEqual(call_kwargs["model"], "gemini-3.5-flash")
        self.assertEqual(call_kwargs["max_tokens"], 8192)
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
        user_msg = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "user")
        policy_section = user_msg.split("<policy_text>")[-1].split("</policy_text>")[0].strip()
        # Full text was passed without condensing
        self.assertNotIn("MIDDLE SECTIONS CONDENSED", policy_section)
        self.assertEqual(len(policy_section), len(full_text))

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "openai")
    @patch("services.rag_auditor.LLM_API_KEY", "fake-key")
    @patch("openai.OpenAI")
    def test_gemini_503_immediate_failover_to_secondary_gemini_model_with_full_text(self, mock_openai_cls):
        """Verifies 503 on primary model fails over immediately to secondary 1M-context model with FULL text and zero sleep."""
        from services.rag_auditor import _audit_with_llm
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        err_503 = Exception("Error code: 503 - High Demand")
        setattr(err_503, "status_code", 503)

        mock_success = MagicMock()
        mock_success.choices = [
            MagicMock(message=MagicMock(content='{"overall_score": 92, "core_criteria": [], "violations": []}'))
        ]

        # Call 1: gemini-3.5-flash -> 503
        # Call 2: immediately gemini-3.5-flash-lite -> SUCCESS (zero sleep delay!)
        mock_client.chat.completions.create.side_effect = [err_503, mock_success]
        mock_openai_cls.return_value = mock_client

        # 42,000 char document
        full_text = ("Kabarak University Detailed Privacy Policy Statement. " * 800).strip()
        self.assertTrue(len(full_text) > 40000)

        result = _audit_with_llm(full_text, [], "Kabarak University", "kabarak.ac.ke")

        self.assertEqual(result["overall_score"], 92)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

        # Call 1 was gemini-3.5-flash
        call1_kwargs = mock_client.chat.completions.create.call_args_list[0][1]
        self.assertEqual(call1_kwargs["model"], "gemini-3.5-flash")
        self.assertEqual(call1_kwargs["max_tokens"], 8192)
        self.assertEqual(call1_kwargs["response_format"], {"type": "json_object"})

        # Call 2 was gemini-3.5-flash-lite
        call2_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        self.assertEqual(call2_kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(call2_kwargs["max_tokens"], 8192)
        self.assertEqual(call2_kwargs["response_format"], {"type": "json_object"})

        # Crucial: verify full 42,000+ chars were passed without any condensing!
        user_msg = next(m["content"] for m in call2_kwargs["messages"] if m["role"] == "user")
        policy_section = user_msg.split("<policy_text>")[-1].split("</policy_text>")[0].strip()
        self.assertNotIn("MIDDLE SECTIONS CONDENSED", policy_section)
        self.assertEqual(len(policy_section), len(full_text))

    @patch("services.rag_auditor.LLM_API_KEY", "fake-key")
    @patch("openai.OpenAI")
    def test_no_duplicate_placeholder_when_section_29_present(self, mock_openai_cls):
        """Verifies placeholder note is merged into existing Section 29 / Controller Identity gap without creating a duplicate."""
        from services.rag_auditor import _audit_with_llm
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="""{
                "overall_score": 70,
                "core_criteria": [{"key": "data_controller_identity", "status": "FAIL", "score": 0}],
                "violations": [
                    {
                        "id": 1,
                        "title": "Incomplete Data Controller Identity Disclosures",
                        "severity": "High",
                        "citation": "Kenya Data Protection Act, 2019, Section 29",
                        "explanation": "Controller physical address is omitted.",
                        "recommended_fix": "Provide registered office address."
                    }
                ]
            }"""))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        text_with_brackets = "Privacy Notice [Insert Email Address] [Physical Address]. Contact us."
        report = _audit_with_llm(text_with_brackets, [], "Test Org")

        # Must have only 1 violation, not 2
        self.assertEqual(len(report["violations"]), 1)
        v = report["violations"][0]
        self.assertIn("Section 29", v["citation"])
        self.assertIn("placeholders", v["explanation"].lower())


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

    @patch("app.fetch_audit_report")
    def test_view_report_renders_without_undefined_error(self, mock_fetch):
        """Verifies GET /report/<id> renders 200 without Jinja2 UndefinedError for full_report."""
        mock_fetch.return_value = {
            "id": "mock-uuid-test",
            "target_url_or_name": "Test Organization",
            "risk_level": "Medium",
            "is_paid": False,
            "mpesa_receipt_number": None,
            "teaser_summary": {"entity_name": "Test Organization"},
            "full_report": {
                "entity_name": "Test Organization",
                "overall_score": 75,
                "executive_summary": "Test summary",
                "violations": [
                    {
                        "id": 1,
                        "title": "Missing DPO Contact",
                        "severity": "High",
                        "citation": "Section 24",
                        "explanation": "No contact details.",
                        "recommended_fix": "Appoint DPO"
                    }
                ]
            },
            "compliant_rewrite": ""
        }
        response = self.client.get("/report/mock-uuid-test")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Test Organization", content)

    @patch("services.rag_auditor.GEMINI_API_KEY", "test-gemini-key")
    @patch("services.rag_auditor.LLM_API_KEY", "test-groq-key")
    @patch("services.rag_auditor.LLM_PROVIDER", "openai")
    def test_llm_gateway_timeout_failover(self, *mocks):
        """Verifies LLMGateway catches APITimeoutError and fails over to subsequent candidates."""
        from services.llm_gateway import LLMGateway
        from openai import APITimeoutError
        gateway = LLMGateway(groq_api_key="test-groq", gemini_api_key="test-gemini")

        call_count = 0
        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First model times out
                raise APITimeoutError("Request timed out after 15.0s")
            # Fallback model succeeds
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps({"overall_score": 80, "violations": []})
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = mock_create

            long_doc = "A" * 10000
            result, model = gateway.generate_json("System prompt", "User prompt", document_text=long_doc)
            self.assertEqual(result["overall_score"], 80)
            self.assertGreaterEqual(call_count, 2)
            call_kwargs = mock_openai_cls.call_args[1]
            self.assertEqual(call_kwargs.get("timeout"), 15.0)
            self.assertEqual(call_kwargs.get("max_retries"), 0)


if __name__ == "__main__":
    unittest.main()
