"""
Safaricom Daraja API integration for Lipa Na M-Pesa Online (STK Push).
Handles OAuth authentication, timestamp/password generation, STK push initiation,
and callback verification for Kenyan ODPC Compliance Audit report unlocking.
"""

import os
import base64
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Daraja API Endpoints
SANDBOX_BASE_URL = "https://sandbox.safaricom.co.ke"
PRODUCTION_BASE_URL = "https://api.safaricom.co.ke"

# Configuration
MPESA_ENV = os.getenv("MPESA_ENVIRONMENT", "sandbox").strip().lower()
CONSUMER_KEY = os.getenv("MPESA_CONSUMER_KEY", "").strip()
CONSUMER_SECRET = os.getenv("MPESA_CONSUMER_SECRET", "").strip()
PASSKEY = os.getenv(
    "MPESA_PASSKEY", 
    "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919"  # Default sandbox passkey
).strip()
SHORTCODE = os.getenv("MPESA_SHORTCODE", "174379").strip()
REPORT_PRICE = int(os.getenv("REPORT_UNLOCK_PRICE_KES", "1000"))


def get_callback_url() -> str:
    """
    Returns the M-Pesa STK push callback URL dynamically from environment variables.
    Ensures no hardcoded localhost or ngrok URLs are used in production.
    Automatically ensures the path ends with /api/mpesa/callback.
    """
    url = os.getenv("MPESA_CALLBACK_URL", "").strip()
    if not url:
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
        if railway_domain:
            url = f"https://{railway_domain}/api/mpesa/callback"
        else:
            url = "https://web-production-6eedea49.up.railway.app/api/mpesa/callback"

    if url and not url.endswith("/api/mpesa/callback"):
        url = url.rstrip("/") + "/api/mpesa/callback"

    return url

# In-memory access token cache: (token, expiry_timestamp)
_TOKEN_CACHE: Tuple[Optional[str], float] = (None, 0.0)


def get_base_url() -> str:
    """Returns the base Daraja URL based on MPESA_ENVIRONMENT."""
    return PRODUCTION_BASE_URL if MPESA_ENV == "production" else SANDBOX_BASE_URL


def format_phone_number(phone: str) -> str:
    """
    Normalizes a Kenyan phone number into the required Safaricom format: 2547XXXXXXXX or 2541XXXXXXXX.
    Accepts:
        - 0712345678 -> 254712345678
        - 0112345678 -> 254112345678
        - +254712345678 -> 254712345678
        - 254712345678 -> 254712345678
        - 712345678 -> 254712345678
    """
    clean = re.sub(r"[\s\-\+\(\)]", "", str(phone or ""))
    
    if clean.startswith("0"):
        clean = "254" + clean[1:]
    elif clean.startswith("7") or clean.startswith("1"):
        clean = "254" + clean
        
    if not re.match(r"^254[71]\d{8}$", clean):
        raise ValueError(
            f"Invalid Kenyan phone number format: '{phone}'. "
            "Please provide a valid Safaricom number (e.g. 0712345678 or 254712345678)."
        )
    return clean


def generate_daraja_token() -> str:
    """
    Generates or retrieves a cached OAuth access token from Safaricom Daraja.
    
    Returns:
        str: Valid Bearer access token.
    """
    global _TOKEN_CACHE
    now = datetime.now().timestamp()
    cached_token, expiry = _TOKEN_CACHE
    
    # Return cached token if valid with 60-second safety margin
    if cached_token and now < (expiry - 60):
        return cached_token

    if not CONSUMER_KEY or not CONSUMER_SECRET:
        logger.warning("MPESA_CONSUMER_KEY or MPESA_CONSUMER_SECRET is not configured.")
        raise ValueError(
            "M-Pesa credentials not configured. Please set MPESA_CONSUMER_KEY and "
            "MPESA_CONSUMER_SECRET in your .env file."
        )

    auth_url = f"{get_base_url()}/oauth/v1/generate?grant_type=client_credentials"
    auth_bytes = f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode("utf-8")
    auth_header = base64.b64encode(auth_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_header}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(auth_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 3599))
        
        _TOKEN_CACHE = (token, now + expires_in)
        logger.info("Successfully obtained new Daraja OAuth token (expires in %ds)", expires_in)
        return token
    except requests.exceptions.RequestException as e:
        logger.error("Failed to authenticate with Safaricom Daraja: %s", e)
        raise RuntimeError(f"Daraja OAuth authentication failed: {e}") from e


def generate_stk_password(shortcode: str, passkey: str, timestamp: str) -> str:
    """Generates base64 encoded password for Daraja STK Push."""
    data_to_encode = f"{shortcode}{passkey}{timestamp}".encode("utf-8")
    return base64.b64encode(data_to_encode).decode("utf-8")


def initiate_stk_push(
    phone_number: str, 
    amount: int = REPORT_PRICE, 
    account_ref: str = "ODPC-Audit",
    audit_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Triggers an M-Pesa STK Push (Lipa Na M-Pesa Online) to the user's phone.
    
    Args:
        phone_number: Kenyan mobile number (07XX / 01XX / 2547XX).
        amount: Transaction amount in KES (default: 1000).
        account_ref: Reference displayed on customer's phone.
        audit_id: UUID of the audit report to link in callback.
        
    Returns:
        Dict with keys:
            - success (bool)
            - checkout_request_id (str)
            - merchant_request_id (str)
            - customer_message (str)
            - response_code (str)
            - error (Optional[str])
    """
    try:
        formatted_phone = format_phone_number(phone_number)
    except ValueError as ve:
        return {
            "success": False,
            "error": str(ve),
            "customer_message": str(ve)
        }

    # If credentials are explicitly missing in development, provide a graceful simulation response
    if not CONSUMER_KEY or not CONSUMER_SECRET:
        mock_checkout_id = f"ws_CO_{datetime.now().strftime('%d%m%Y%H%M%S')}_{audit_id[:8] if audit_id else 'DEMO'}"
        logger.warning(
            "Daraja credentials not configured. Initiating simulated sandbox STK push: %s",
            mock_checkout_id
        )
        return {
            "success": True,
            "checkout_request_id": mock_checkout_id,
            "merchant_request_id": f"MOCK_MERCHANT_{audit_id[:8] if audit_id else 'TEST'}",
            "customer_message": (
                f"[SANDBOX SIMULATION] STK push sent to {formatted_phone} for KES {amount}. "
                "Use the 'Simulate Payment' button to confirm payment without live Safaricom credentials."
            ),
            "response_code": "0",
            "is_simulation": True
        }

    try:
        access_token = generate_daraja_token()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password = generate_stk_password(SHORTCODE, PASSKEY, timestamp)

        stk_endpoint = f"{get_base_url()}/mpesa/stkpush/v1/processrequest"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Embed audit_id into account_ref or transaction description for traceability
        short_ref = (account_ref[:12] if len(account_ref) > 12 else account_ref)
        
        callback_url = get_callback_url()
        if not callback_url:
            logger.warning(
                "MPESA_CALLBACK_URL is not set in environment variables. "
                "Safaricom will be unable to reach the callback endpoint."
            )

        payload = {
            "BusinessShortCode": SHORTCODE,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": int(amount),
            "PartyA": formatted_phone,
            "PartyB": SHORTCODE,
            "PhoneNumber": formatted_phone,
            "CallBackURL": callback_url or "https://seraguard.co.ke/api/mpesa/callback",
            "AccountReference": short_ref,
            "TransactionDesc": f"ODPC Audit Unlock {audit_id[:8] if audit_id else ''}"
        }

        logger.info("Sending STK Push to %s for KES %d...", formatted_phone, amount)
        response = requests.post(stk_endpoint, json=payload, headers=headers, timeout=25)
        res_data = response.json()
        
        logger.info("Daraja STK response: %s", res_data)

        response_code = str(res_data.get("ResponseCode", ""))
        if response_code == "0":
            return {
                "success": True,
                "checkout_request_id": res_data.get("CheckoutRequestID"),
                "merchant_request_id": res_data.get("MerchantRequestID"),
                "customer_message": res_data.get("CustomerMessage", "STK push initiated successfully. Please enter your M-Pesa PIN."),
                "response_code": response_code,
                "is_simulation": False
            }
        else:
            error_desc = res_data.get("ResponseDescription", res_data.get("errorMessage", "Daraja rejected request."))
            return {
                "success": False,
                "error": error_desc,
                "customer_message": error_desc,
                "response_code": response_code,
                "is_simulation": False
            }

    except Exception as e:
        logger.exception("Exception initiating Daraja STK Push: %s", e)
        return {
            "success": False,
            "error": str(e),
            "customer_message": f"Network error connecting to Safaricom M-Pesa: {e}",
            "is_simulation": False
        }


def parse_mpesa_callback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses incoming Safaricom Daraja STK Push callback payload.
    
    Args:
        payload: Decoded JSON dictionary from Safaricom POST request.
        
    Returns:
        Dict containing:
            - success (bool): True if ResultCode == 0
            - result_code (int)
            - result_desc (str)
            - checkout_request_id (str)
            - merchant_request_id (str)
            - mpesa_receipt_number (Optional[str])
            - amount (Optional[float])
            - phone_number (Optional[str])
    """
    body = payload.get("Body", {})
    stk_callback = body.get("stkCallback", {})
    
    result_code = stk_callback.get("ResultCode")
    result_desc = stk_callback.get("ResultDesc", "No description provided.")
    checkout_id = stk_callback.get("CheckoutRequestID")
    merchant_id = stk_callback.get("MerchantRequestID")

    is_success = (result_code == 0)

    receipt_number = None
    amount = None
    phone_number = None

    if is_success:
        callback_items = stk_callback.get("CallbackMetadata", {}).get("Item", [])
        for item in callback_items:
            name = item.get("Name")
            value = item.get("Value")
            if name == "MpesaReceiptNumber":
                receipt_number = str(value)
            elif name == "Amount":
                amount = float(value)
            elif name == "PhoneNumber":
                phone_number = str(value)

    return {
        "success": is_success,
        "result_code": result_code,
        "result_desc": result_desc,
        "checkout_request_id": checkout_id,
        "merchant_request_id": merchant_id,
        "mpesa_receipt_number": receipt_number,
        "amount": amount,
        "phone_number": phone_number
    }


def query_stk_status(checkout_request_id: str) -> Dict[str, Any]:
    """
    Queries Safaricom Daraja STK Push Query API to verify status of an STK transaction.
    Protects against spoofed callback attacks by querying Daraja directly.

    Args:
        checkout_request_id: CheckoutRequestID received during initiate_stk_push.

    Returns:
        Dict containing success, result_code, result_desc.
    """
    if not CONSUMER_KEY or not CONSUMER_SECRET:
        # Development / sandbox mock mode: return success if mock checkout ID
        if checkout_request_id and "ws_CO_" in checkout_request_id:
            return {"success": True, "result_code": 0, "result_desc": "Mock verified in dev/sandbox mode."}
        return {"success": False, "result_code": -1, "result_desc": "M-Pesa credentials not configured."}

    try:
        access_token = generate_daraja_token()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password = generate_stk_password(SHORTCODE, PASSKEY, timestamp)

        query_endpoint = f"{get_base_url()}/mpesa/stkpushquery/v1/query"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "BusinessShortCode": SHORTCODE,
            "Password": password,
            "Timestamp": timestamp,
            "CheckoutRequestID": checkout_request_id
        }

        logger.info("Querying Daraja STK status for CheckoutRequestID %s...", checkout_request_id)
        resp = requests.post(query_endpoint, json=payload, headers=headers, timeout=20)
        data = resp.json()
        logger.info("Daraja STK status query response: %s", data)

        res_code = str(data.get("ResultCode", ""))
        is_success = (res_code == "0")

        return {
            "success": is_success,
            "result_code": data.get("ResultCode"),
            "result_desc": data.get("ResultDesc", "No description returned from Daraja."),
            "response_code": data.get("ResponseCode")
        }
    except Exception as e:
        logger.exception("Failed to query Daraja STK status: %s", e)
        return {
            "success": False,
            "result_code": -1,
            "result_desc": f"Network error during Daraja query: {e}"
        }
