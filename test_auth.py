#!/usr/bin/env python3
"""End-to-end test script for JWT authentication."""

import requests

BASE_URL = "http://localhost:8000"

def test_auth_flow():
    print("=" * 50)
    print("Enterprise RAG - Auth Flow Test")
    print("=" * 50)

    # 1. Health check
    print("\n[1] Health Check...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {resp.json()}")

    # 2. Register
    print("\n[2] Registering new user...")
    user_data = {"email": "testuser@example.com", "password": "testpass123"}
    resp = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    if resp.status_code == 201:
        print(f"    Success! User created: {resp.json()['email']}")
    elif resp.status_code == 400:
        print(f"    User already exists (that's OK for re-runs)")
    else:
        print(f"    Error: {resp.status_code} - {resp.text}")

    # 3. Login
    print("\n[3] Logging in...")
    login_data = {"username": "testuser@example.com", "password": "testpass123"}
    resp = requests.post(f"{BASE_URL}/auth/login", data=login_data)
    if resp.status_code != 200:
        print(f"    Login failed: {resp.status_code} - {resp.text}")
        return

    token = resp.json()["access_token"]
    print(f"    Success! Token: {token[:50]}...")

    headers = {"Authorization": f"Bearer {token}"}

    # 4. Get current user
    print("\n[4] Getting current user info...")
    resp = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    print(f"    Status: {resp.status_code}")
    print(f"    User: {resp.json()}")

    # 5. Test protected route without token
    print("\n[5] Testing protected route WITHOUT token...")
    resp = requests.get(f"{BASE_URL}/documents")
    print(f"    Status: {resp.status_code} (expected 401)")
    assert resp.status_code == 401, "Should be unauthorized!"
    print(f"    Correctly rejected!")

    # 6. Test protected route with token
    print("\n[6] Testing protected route WITH token...")
    resp = requests.get(f"{BASE_URL}/documents", headers=headers)
    print(f"    Status: {resp.status_code}")
    print(f"    Documents: {resp.json()}")

    # 7. Test with invalid token
    print("\n[7] Testing with INVALID token...")
    bad_headers = {"Authorization": "Bearer invalid_token_here"}
    resp = requests.get(f"{BASE_URL}/documents", headers=bad_headers)
    print(f"    Status: {resp.status_code} (expected 401)")
    assert resp.status_code == 401, "Should reject invalid token!"
    print(f"    Correctly rejected!")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_auth_flow()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server.")
        print("Make sure the FastAPI server is running:")
        print("  uvicorn app.main:app --reload")
