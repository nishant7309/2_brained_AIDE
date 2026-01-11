"""
Simple test to verify Gemini API is working with AIDE
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

# Test just the model query, not the full experiment
from aide.backend import query
from aide.utils.config import _load_cfg

cfg = _load_cfg(use_cli_args=False)

print("=" * 60)
print("Testing Gemini API Connection")
print("=" * 60)
print(f"Planner model: {cfg.agent.planner.model}")
print(f"Coder model: {cfg.agent.coder.model}")
print("=" * 60)

# Simple test query
print("\nSending test query to Gemini...")
response = query(
    system_message="You are a helpful assistant.",
    user_message="Say 'Hello from Gemini!' and nothing else.",
    model=cfg.agent.coder.model,
    temperature=0.0,
)

print(f"\nResponse: {response}")
print("\n" + "=" * 60)
print("SUCCESS! Gemini API is working!")
print("=" * 60)
