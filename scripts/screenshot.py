"""Automated screenshot script — captures all 5 tabs of UrbanGraph-SG dashboard.

Usage: python scripts/screenshot.py
Requirements: pip install playwright, Neo4j + Streamlit running locally
"""

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

PROJECT_ROOT = Path(__file__).parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
APP_URL = "http://localhost:8502"


def capture():
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2,  # retina-quality
        )
        page = context.new_page()

        print("Opening app...")
        page.goto(APP_URL, timeout=60000, wait_until="networkidle")
        time.sleep(3)

        # Tab 0: Explore — already visible
        print("Tab 0: Explore")
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_explore.png"), full_page=False)
        time.sleep(1)

        # Tab 1: Query — click, then ask a preset question
        print("Tab 1: Query")
        page.click('button[role="tab"]:has-text("💬 Query")')
        time.sleep(2)
        # Click first preset button
        try:
            page.click("button:has-text('How many MRT stations are there in total?')", timeout=5000)
            time.sleep(5)  # wait for LLM response
        except PwTimeout:
            print("  (no preset clicked — showing default Query tab)")
        page.screenshot(path=str(SCREENSHOTS_DIR / "02_query.png"), full_page=False)
        time.sleep(1)

        # Tab 2: Analytics
        print("Tab 2: Analytics")
        page.click('button[role="tab"]:has-text("📊 Analytics")')
        time.sleep(3)
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_analytics.png"), full_page=True)
        time.sleep(1)

        # Tab 3: Graph ML
        print("Tab 3: Graph ML")
        page.click('button[role="tab"]:has-text("🔬 Graph ML")')
        time.sleep(2)
        page.screenshot(path=str(SCREENSHOTS_DIR / "04_graph_ml.png"), full_page=False)
        time.sleep(1)

        # Tab 4: Report
        print("Tab 4: Report")
        page.click('button[role="tab"]:has-text("📋 Report")')
        time.sleep(3)
        page.screenshot(path=str(SCREENSHOTS_DIR / "05_report.png"), full_page=True)
        time.sleep(1)

        browser.close()

    print(f"\nDone! {len(list(SCREENSHOTS_DIR.glob('*.png')))} screenshots saved to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    capture()
