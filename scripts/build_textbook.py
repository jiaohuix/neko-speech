#!/usr/bin/env python3
"""
Complete textbook build script for neko-speech.
Generates combined markdown, PDF (via pandoc/xelatex), and HTML.
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
CHAPTERS_DIR = PROJECT_ROOT / "chapters"

# Chapter order (explicit to get the right sequence)
CHAPTER_ORDER = [
    "ch01_audio_fundamentals",
    "ch02_tacotron",
    "ch03_wavenet",
    "ch04_fastspeech",
    "ch05_vits",
    "ch06_neural_codec",
    "ch07_valle",
    "ch08_modern_models",
    "ch09_gpt_sovits",
    "ch10_voxcpm",
    "ch11_minimind_o",
]

# Chapter display names
CHAPTER_NAMES = {
    "ch01_audio_fundamentals": "Audio Fundamentals",
    "ch02_tacotron": "Tacotron 2",
    "ch03_wavenet": "WaveNet",
    "ch04_fastspeech": "FastSpeech",
    "ch05_vits": "VITS",
    "ch06_neural_codec": "Neural Codec",
    "ch07_valle": "VALL-E",
    "ch08_modern_models": "Modern TTS Models",
    "ch09_gpt_sovits": "GPT-SoVITS",
    "ch10_voxcpm": "VoxCPM",
    "ch11_minimind_o": "MiniMind-O",
}


def find_readme(ch_dir_name):
    """Find the README.md for a chapter, checking subdirectories."""
    ch_path = CHAPTERS_DIR / ch_dir_name
    readme = ch_path / "README.md"
    if readme.exists():
        return readme, ch_path
    # Check code/ subdirectory (ch08 case)
    readme = ch_path / "code" / "README.md"
    if readme.exists():
        return readme, ch_path / "code"
    return None, None


def fix_image_paths(content, base_dir):
    """Convert relative image paths to absolute paths."""
    def replace_path(match):
        alt = match.group(1)
        path = match.group(2)
        if path.startswith("http://") or path.startswith("https://"):
            return match.group(0)
        # Make path absolute relative to project root
        abs_path = (base_dir / path).resolve()
        return f"![{alt}]({abs_path})"

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_path, content)


def create_combined_markdown():
    """Create the master combined markdown document."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BUILD_DIR / "neko-speech-complete.md"

    with open(output_path, "w", encoding="utf-8") as out:
        # YAML front matter
        out.write("---\n")
        out.write('title: "从 WaveNet 到 Omni：语音合成教科书"\n')
        out.write('subtitle: "从零开始用 PyTorch 构建猫娘语音助手"\n')
        out.write(f'date: "{datetime.now().strftime("%Y-%m-%d")}"\n')
        out.write('author: "Neko Speech Team"\n')
        out.write("---\n\n")

        # Title page
        out.write("\\begin{center}\n")
        out.write("\\vspace*{3cm}\n")
        out.write("{\\Huge\\bfseries 从 WaveNet 到 Omni\\\\[0.5cm] 语音合成教科书}\n\n")
        out.write("\\vspace{1cm}\n")
        out.write("{\\Large 从零开始用 PyTorch 构建猫娘语音助手}\n\n")
        out.write("\\vspace{2cm}\n")
        out.write("{\\large Neko Speech Team}\n\n")
        out.write(f"{{\\large {datetime.now().strftime('%Y-%m-%d')}}}\n\n")
        out.write("\\vspace{1cm}\n")
        out.write("\\textit{喵\~{} 和猫娘一起学语音合成吧!}\n")
        out.write("\\end{center}\n\n")
        out.write("\\newpage\n\n")

        # Table of contents
        out.write("\\tableofcontents\n\n")
        out.write("\\newpage\n\n")

        # Chapters
        found = 0
        for ch_name in CHAPTER_ORDER:
            readme, base_dir = find_readme(ch_name)
            if readme is None:
                print(f"  [SKIP] {ch_name}: no README.md found")
                continue

            print(f"  [OK] {ch_name}: {readme}")
            found += 1

            content = readme.read_text(encoding="utf-8")

            # Fix image paths
            content = fix_image_paths(content, base_dir)

            # Write chapter header with page break
            display_name = CHAPTER_NAMES.get(ch_name, ch_name)
            out.write(f"<!-- Chapter: {ch_name} -->\n\n")
            out.write(content)
            out.write("\n\n\\newpage\n\n")

        print(f"\n  Total chapters included: {found}")

    return output_path


def generate_pdf(markdown_path, output_path):
    """Generate PDF using pandoc + xelatex."""
    print("\n=== Generating PDF ===")

    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(output_path),
        "--pdf-engine=xelatex",
        "-V", "mainfont=Noto Serif CJK SC",
        "-V", "sansfont=Noto Sans CJK SC",
        "-V", "monofont=Noto Sans Mono CJK SC",
        "-V", "CJKmainfont=Noto Sans CJK SC",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=11pt",
        "-V", "linkcolor=blue",
        "-V", "urlcolor=blue",
        "-V", "citecolor=blue",
        "--toc",
        "--toc-depth=2",
        "-N",
        "--highlight-style=tango",
        "--resource-path=" + str(PROJECT_ROOT),
    ]

    print(f"  Command: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        size = os.path.getsize(output_path)
        print(f"  PDF generated: {output_path} ({size:,} bytes)")
        return True
    else:
        print(f"  PDF generation failed (exit {result.returncode})")
        if result.stderr:
            # Write full error to log
            log_path = BUILD_DIR / "pdf_error.log"
            log_path.write_text(result.stderr, encoding="utf-8")
            # Print last 30 lines
            lines = result.stderr.strip().split("\n")
            for line in lines[-30:]:
                print(f"    {line}")
            print(f"  Full error log: {log_path}")
        return False


def generate_pdf_weasyprint(markdown_path, output_path):
    """Fallback: generate PDF via weasyprint (markdown -> html -> pdf)."""
    print("\n=== Generating PDF via WeasyPrint (fallback) ===")

    html_path = BUILD_DIR / "neko-speech-temp.html"

    # First generate HTML with pandoc
    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(html_path),
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "--metadata", "title=从 WaveNet 到 Omni：语音合成教科书",
        "--highlight-style=tango",
        f"--css={BUILD_DIR / 'style.css'}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  HTML generation for WeasyPrint failed: {result.stderr[-500:]}")
        return False

    # Then convert to PDF
    try:
        import weasyprint
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(output_path))
        size = os.path.getsize(output_path)
        print(f"  PDF generated via WeasyPrint: {output_path} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  WeasyPrint failed: {e}")
        return False


def generate_html(markdown_path, output_path):
    """Generate standalone HTML."""
    print("\n=== Generating HTML ===")

    # Create CSS file
    css_path = BUILD_DIR / "style.css"
    css_path.write_text(BOOK_CSS, encoding="utf-8")

    cmd = [
        "pandoc",
        str(markdown_path),
        "-o", str(output_path),
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "--metadata", "title=从 WaveNet 到 Omni：语音合成教科书",
        "--highlight-style=tango",
        f"--css={css_path}",
        "--embed-resources",
        "--standalone",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        size = os.path.getsize(output_path)
        print(f"  HTML generated: {output_path} ({size:,} bytes)")
        return True
    else:
        print(f"  HTML generation failed: {result.stderr[-500:]}")
        # Try simpler version without embed-resources
        cmd_simple = [c for c in cmd if c != "--embed-resources"]
        result2 = subprocess.run(cmd_simple, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if result2.returncode == 0:
            size = os.path.getsize(output_path)
            print(f"  HTML generated (no embed): {output_path} ({size:,} bytes)")
            return True
        return False


# CSS for the HTML version
BOOK_CSS = """
body {
    font-family: "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 2em;
    line-height: 1.8;
    color: #333;
    background: #fafafa;
}
h1 {
    color: #d63384;
    border-bottom: 3px solid #d63384;
    padding-bottom: 0.3em;
    font-size: 2em;
}
h2 {
    color: #6f42c1;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}
h3 {
    color: #0d6efd;
}
blockquote {
    border-left: 4px solid #d63384;
    background: #fff0f5;
    padding: 0.5em 1em;
    margin: 1em 0;
    font-style: italic;
}
code {
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}
pre {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.5;
}
pre code {
    background: none;
    padding: 0;
    color: inherit;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th {
    background: #d63384;
    color: white;
}
tr:nth-child(even) {
    background: #f8f8f8;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
#toc {
    background: white;
    border: 1px solid #ddd;
    padding: 1.5em;
    border-radius: 6px;
    margin-bottom: 2em;
}
#toc a {
    text-decoration: none;
    color: #d63384;
}
#toc a:hover {
    text-decoration: underline;
}
a {
    color: #d63384;
}
hr {
    border: none;
    border-top: 2px dashed #ddd;
    margin: 3em 0;
}
"""


def main():
    log_lines = []
    start_time = datetime.now()

    print("=" * 60)
    print("  Neko Speech Textbook Builder")
    print("=" * 60)
    log_lines.append(f"Build started: {start_time.isoformat()}")

    # Step 1: Create combined markdown
    print("\n--- Step 1: Creating combined markdown ---")
    md_path = create_combined_markdown()
    md_size = os.path.getsize(md_path)
    print(f"  Combined markdown: {md_path} ({md_size:,} bytes)")
    log_lines.append(f"Combined markdown: {md_size:,} bytes")

    # Step 2: Generate PDF
    print("\n--- Step 2: Generating PDF ---")
    pdf_path = BUILD_DIR / "neko-speech-textbook.pdf"
    pdf_ok = generate_pdf(md_path, pdf_path)
    if not pdf_ok:
        print("  Trying WeasyPrint fallback...")
        pdf_ok = generate_pdf_weasyprint(md_path, pdf_path)
    log_lines.append(f"PDF: {'OK' if pdf_ok else 'FAILED'}")

    # Step 3: Generate HTML
    print("\n--- Step 3: Generating HTML ---")
    html_path = BUILD_DIR / "neko-speech-textbook.html"
    html_ok = generate_html(md_path, html_path)
    log_lines.append(f"HTML: {'OK' if html_ok else 'FAILED'}")

    # Step 4: Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("  BUILD SUMMARY")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")

    results = []
    for name, path, ok in [
        ("Combined MD", md_path, True),
        ("PDF", pdf_path, pdf_ok),
        ("HTML", html_path, html_ok),
    ]:
        if ok and path.exists():
            size = os.path.getsize(path)
            print(f"  [OK] {name}: {path} ({size:,} bytes)")
            results.append(f"{name}: {path} ({size:,} bytes)")
        else:
            print(f"  [FAIL] {name}: {path}")
            results.append(f"{name}: FAILED")

    log_lines.append(f"Build time: {elapsed:.1f}s")
    log_lines.extend(results)

    # Write log
    log_path = BUILD_DIR / "generation.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n  Log: {log_path}")

    return pdf_ok and html_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
