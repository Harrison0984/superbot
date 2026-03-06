"""Test IMAP connection to Gmail."""

import imaplib
import ssl

host = "imap.gmail.com"
port = 993
username = "vbanglev@gmail.com"
password = "kxsh pxmi wtjv puuf"

print(f"Testing IMAP connection to {host}:{port}...")

# Test with timeout
try:
    context = ssl.create_default_context()
    client = imaplib.IMAP4_SSL(host, port, timeout=30)
    print("Connected!")

    client.login(username, password)
    print(f"Logged in as {username}")

    status, _ = client.select("INBOX")
    print(f"Inbox status: {status}")

    # Search for recent messages
    status, data = client.search(None, "ALL")
    if status == "OK":
        ids = data[0].split()
        print(f"Total messages: {len(ids)}")

    client.logout()
    print("Done!")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
