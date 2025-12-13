from pathlib import Path
import os
import json
from playwright.sync_api import sync_playwright
import time

URL   = "https://www.i2ocr.com/pdf-ocr-urdu"

# EITHER set FOLDER to a directory of PDFs...
FOLDER = r"F:\HHH\HHH\SNA\Books\QuratUlAinHaider\KarEJahanDaraz"
# ...OR leave FOLDER empty and set a single FILE to process just one PDF.cls

FILE   = r""  # e.g., r"C:\path\to\some.pdf"


# ----------------------------
# Resume / checkpoint helpers
# ----------------------------
def load_json(path: Path, default=None):
    """Load JSON from path if it exists; otherwise return default."""
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: Path, data: dict):
    """Save JSON to path (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def ocr_one_pdf(pdf_path: Path):
    """Implements your existing per-PDF logic (A), but saves into:
       <pdf parent>/results/<pdfname>_results/<pdfname>-<pageno>.txt
    """
    assert pdf_path.exists(), f"File not found: {pdf_path}"
    base = pdf_path.stem

    # results root = <folder>/results; then per-PDF subfolder <base>_results
    results_root = pdf_path.parent / "results"
    results_root.mkdir(exist_ok=True)
    results_dir = results_root / f"{base}_results"
    results_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Per-PDF status checkpointing
    # ----------------------------
    status_path = results_dir / "status.json"
    status = load_json(status_path, default={
        "pages_done": [],
        "pages_fail": [],
        "last_page_attempted": 0
    })

    # Detect already completed pages by existing output files
    existing_done = set()
    for txt_file in results_dir.glob(f"{base}-*.txt"):
        try:
            page_str = txt_file.stem.split("-")[-1]
            page_no = int(page_str)
            if page_no > 0:
                existing_done.add(page_no)
        except Exception:
            continue

    # Merge detected done pages into status.json
    pages_done = set(status.get("pages_done", []))
    pages_fail = set(status.get("pages_fail", []))
    pages_done |= existing_done
    pages_fail -= existing_done

    status["pages_done"] = sorted(pages_done)
    status["pages_fail"] = sorted(pages_fail)
    status["last_page_attempted"] = max(
        status.get("last_page_attempted", 0),
        max(existing_done) if existing_done else 0
    )
    save_json(status_path, status)

    # Compute dynamic resume start = FIRST missing page (handles gaps)
    start_page = 1
    for pno in range(1, 22):  # 1..21
        if (results_dir / f"{base}-{pno}.txt").exists():
            continue
        start_page = pno
        break
    start_i = start_page - 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        try:
            page = browser.new_page()
            page.set_default_timeout(200000)

            page.goto(URL, wait_until="domcontentloaded")
            page.set_input_files("#upload_file", str(pdf_path))

            print(f"\n==> Uploaded: {pdf_path.name}")
            print("Waiting for PDF preview to load...")
            try:
                page.wait_for_load_state("networkidle", timeout=100000)
                print("Page reached network idle state")
            except Exception as e:
                print(f"Network idle timeout (ok if still loading): {e}", "WARNING")
            time.sleep(5)
            print("PDF preview loaded.\n")

            # --- A: your existing per-page loop (kept as-is, just writes to results_dir) ---
            for i in range(start_i, 21):
                page_no = i + 1
                out_txt = results_dir / f"{base}-{page_no}.txt"

                # Skip any page whose output already exists
                if out_txt.exists():
                    if page_no not in pages_done:
                        pages_done.add(page_no)
                        status["pages_done"] = sorted(pages_done)
                        status["pages_fail"] = sorted(pages_fail)
                        status["last_page_attempted"] = max(status.get("last_page_attempted", 0), page_no)
                        save_json(status_path, status)
                    print(f"⏭️  Skipping page {page_no} (already exists): {out_txt.name}")
                    continue

                s = f"#img-{page_no}"

                try:
                    if page.locator(s).count() == 0:
                        print(f"No thumbnail {s}; stopping.")
                        break

                    page.locator(s).scroll_into_view_if_needed()
                    page.locator(s).click()

                    page.evaluate("""
                        () => {
                            const el = document.querySelector('#ocrTextBox');
                            if (!el) return;
                            if (el.tagName === 'TEXTAREA') el.value = '';
                            else el.textContent = '';
                        }
                    """)

                    page.wait_for_function(
                        "document.querySelector('div.submit_img_ocr:not(.disabled)') !== null"
                    )
                    page.locator("div.submit_img_ocr:not(.disabled)").first.click()

                    cb = page.get_by_role("checkbox", name="I'm not a robot")
                    page.wait_for_timeout(200)
                    try:
                        cb.set_checked(True, timeout=15000)
                    except Exception:
                        try:
                            page.locator('label[for^="altcha_checkbox_"]').click(timeout=15000)
                        except Exception:
                            real_cb = page.locator('input[type="checkbox"][id^="altcha_checkbox_"]')
                            real_cb.wait_for(state="attached", timeout=15000)
                            real_cb.check(force=True, timeout=15000)

                    is_on = page.locator('input[type="checkbox"][id^="altcha_checkbox_"]').first.is_checked()
                    assert is_on, "Checkbox didn't end up checked — selector may need tightening."

                    page.wait_for_function("""
                    () => {
                        const el = document.querySelector('#ocrTextBox');
                        if (!el) return false;
                        const v = el.tagName === 'TEXTAREA' ? el.value : el.textContent;
                        return v && v.trim().length > 0;
                    }
                    """, timeout=120000)

                    box = page.locator("#ocrTextBox")
                    try:
                        text = box.input_value().strip()
                    except Exception:
                        text = (box.text_content() or "").strip()

                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"✅ OCR done. Saved: {out_txt}")

                    pages_done.add(page_no)
                    if page_no in pages_fail:
                        pages_fail.remove(page_no)
                    status["pages_done"] = sorted(pages_done)
                    status["pages_fail"] = sorted(pages_fail)
                    status["last_page_attempted"] = page_no
                    save_json(status_path, status)

                except Exception as e:
                    pages_fail.add(page_no)
                    status["pages_done"] = sorted(pages_done)
                    status["pages_fail"] = sorted(pages_fail)
                    status["last_page_attempted"] = page_no
                    save_json(status_path, status)

                    print(f"❌ Page {page_no} failed: {e}")

                    msg = str(e)
                    if "Target page, context or browser has been closed" in msg or "TargetClosedError" in msg:
                        print("⚠️ Browser/page closed unexpectedly. Exiting this PDF so resume can continue later.")
                        break
                    continue

            # After finishing forward pass, retry any failed pages once
            if pages_fail:
                print(f"\nRetrying failed pages: {sorted(pages_fail)}")

            for pno in sorted(list(pages_fail)):
                out_txt = results_dir / f"{base}-{pno}.txt"
                if out_txt.exists():
                    pages_fail.discard(pno)
                    continue

                s = f"#img-{pno}"

                try:
                    if page.locator(s).count() == 0:
                        print(f"No thumbnail {s}; skipping retry.")
                        continue

                    # ---- SAME OCR STEPS AS BEFORE (copy/paste unchanged) ----
                    page.locator(s).scroll_into_view_if_needed()
                    page.locator(s).click()

                    page.evaluate("""
                        () => {
                            const el = document.querySelector('#ocrTextBox');
                            if (!el) return;
                            if (el.tagName === 'TEXTAREA') el.value = '';
                            else el.textContent = '';
                        }
                    """)

                    page.wait_for_function(
                        "document.querySelector('div.submit_img_ocr:not(.disabled)') !== null"
                    )
                    page.locator("div.submit_img_ocr:not(.disabled)").first.click()

                    cb = page.get_by_role("checkbox", name="I'm not a robot")
                    page.wait_for_timeout(200)
                    try:
                        cb.set_checked(True, timeout=15000)
                    except Exception:
                        try:
                            page.locator('label[for^="altcha_checkbox_"]').click(timeout=15000)
                        except Exception:
                            real_cb = page.locator('input[type="checkbox"][id^="altcha_checkbox_"]')
                            real_cb.wait_for(state="attached", timeout=15000)
                            real_cb.check(force=True, timeout=15000)

                    is_on = page.locator('input[type="checkbox"][id^="altcha_checkbox_"]').first.is_checked()
                    assert is_on, "Checkbox didn't end up checked — selector may need tightening."

                    page.wait_for_function("""
                    () => {
                        const el = document.querySelector('#ocrTextBox');
                        if (!el) return false;
                        const v = el.tagName === 'TEXTAREA' ? el.value : el.textContent;
                        return v && v.trim().length > 0;
                    }
                    """, timeout=120000)

                    box = page.locator("#ocrTextBox")
                    try:
                        text = box.input_value().strip()
                    except Exception:
                        text = (box.text_content() or "").strip()

                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"✅ OCR retry done. Saved: {out_txt}")

                    pages_done.add(pno)
                    pages_fail.discard(pno)
                    status["pages_done"] = sorted(pages_done)
                    status["pages_fail"] = sorted(pages_fail)
                    status["last_page_attempted"] = pno
                    save_json(status_path, status)

                except Exception as e:
                    print(f"❌ Retry page {pno} failed again: {e}")
                    status["pages_fail"] = sorted(pages_fail)
                    status["last_page_attempted"] = pno
                    save_json(status_path, status)

        finally:
            browser.close()

def main():
    if FOLDER and FOLDER.strip():
        folder = Path(FOLDER)
        assert folder.exists() and folder.is_dir(), f"Folder not found: {folder}"

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in: {folder}")
            return

        # Create the top-level results folder once
        results_root = (folder / "results")
        results_root.mkdir(exist_ok=True)

        # ----------------------------
        # Global progress checkpointing
        # ----------------------------
        progress_path = results_root / "_progress.json"
        progress = load_json(progress_path, default={"current_pdf_index": 0})
        current_pdf_index = int(progress.get("current_pdf_index", 0) or 0)
        if current_pdf_index < 0:
            current_pdf_index = 0
        if current_pdf_index >= len(pdfs):
            current_pdf_index = 0

        print(f"Found {len(pdfs)} PDFs in {folder}.")
        for pdf_index in range(current_pdf_index, len(pdfs)):
            pdf = pdfs[pdf_index]

            # Update progress BEFORE processing each PDF
            save_json(progress_path, {"current_pdf_index": pdf_index})

            print(f"\n=== [{pdf_index+1}/{len(pdfs)}] Processing {pdf.name} ===")
            ocr_one_pdf(pdf)

        # Mark completion
        save_json(progress_path, {"current_pdf_index": len(pdfs)})

        print("\nAll done.")
    else:
        # Single-file mode for convenience
        file_path = Path(FILE)
        assert file_path.exists(), f"File not found: {file_path}"
        ocr_one_pdf(file_path)


if __name__ == "__main__":
    main()