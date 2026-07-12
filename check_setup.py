"""
Standalone setup checker. Run this from the project root:

    python check_setup.py

It checks each piece needed for the Azure OpenAI connection to work,
one at a time, and tells you exactly which step failed and why.
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("STEP 1: Checking .env file location")
print("=" * 60)
project_root = Path(__file__).resolve().parent
env_path = project_root / ".env"
if env_path.exists():
    print(f"OK: Found .env at {env_path}")
else:
    print(f"FAIL: No .env file at {env_path}")
    print("      Create it there (not inside .venv) before continuing.")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Checking python-dotenv is installed")
print("=" * 60)
try:
    from dotenv import load_dotenv
    print("OK: python-dotenv is installed")
except ImportError:
    print("FAIL: python-dotenv is not installed.")
    print("      Run: pip install -r requirements.txt")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 3: Loading .env and checking required variables")
print("=" * 60)
load_dotenv(env_path, override=True)
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]
missing = []
for name in required_vars:
    value = os.getenv(name)
    if value:
        shown = value if name != "AZURE_OPENAI_API_KEY" else (value[:4] + "..." + value[-4:] if len(value) > 8 else "****")
        print(f"OK: {name} = {shown}")
    else:
        print(f"FAIL: {name} is missing or empty")
        missing.append(name)

if missing:
    print()
    print(f"Fix your .env file: the following are missing: {', '.join(missing)}")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 4: Checking openai package is installed")
print("=" * 60)
try:
    from openai import AzureOpenAI
    print("OK: openai package is installed")
except ImportError:
    print("FAIL: openai package is not installed.")
    print("      Run: pip install -r requirements.txt")
    sys.exit(1)

print()
print("=" * 60)
print("STEP 5: Attempting a real Azure OpenAI API call")
print("=" * 60)
try:
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    print(f"Sending a minimal test request to deployment '{deployment}'...")
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        max_completion_tokens=5,
    )
    print("SUCCESS! The API call worked.")
    print("Model replied:", response.choices[0].message.content)
except Exception as exc:
    print("FAIL: The API call raised an error.")
    print()
    print("Full error message below - copy this whole block back to Claude:")
    print("-" * 60)
    print(f"{type(exc).__name__}: {exc}")
    print("-" * 60)
    sys.exit(1)

print()
print("=" * 60)
print("ALL CHECKS PASSED - your Azure OpenAI setup is working correctly.")
print("=" * 60)