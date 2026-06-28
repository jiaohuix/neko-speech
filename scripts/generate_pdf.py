#!/usr/bin/env python3
"""
PDF 生成脚本
将所有章节合并为一本完整的 PDF 教科书
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def check_pandoc():
    """检查 pandoc 是否可用"""
    try:
        result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_pandoc():
    """安装 pandoc"""
    print("正在安装 pandoc...")
    os.system("echo '1' | sudo -S apt-get install -y pandoc texlive-xetex texlive-lang-chinese")


def collect_chapters():
    """收集所有章节的 README.md"""
    chapters_dir = Path("chapters")
    chapter_files = []

    for ch_dir in sorted(chapters_dir.iterdir()):
        if ch_dir.is_dir() and ch_dir.name.startswith("ch"):
            readme = ch_dir / "README.md"
            if readme.exists():
                chapter_files.append(readme)
                print(f"✓ 找到: {readme}")
            else:
                print(f"✗ 缺失: {readme}")

    return chapter_files


def generate_pdf(output_file="neko-speech-textbook.pdf"):
    """生成 PDF"""
    if not check_pandoc():
        install_pandoc()

    # 收集所有章节
    chapters = collect_chapters()
    if not chapters:
        print("❌ 没有找到任何章节")
        return False

    # 创建合并的 markdown
    combined = Path("build/combined.md")
    combined.parent.mkdir(exist_ok=True)

    with open(combined, "w") as out:
        # 标题页
        out.write("---\n")
        out.write('title: "从 WaveNet 到 Omni：语音合成教科书"\n')
        out.write('subtitle: "从零开始用 PyTorch 构建猫娘语音助手"\n')
        out.write(f'date: "{datetime.now().strftime("%Y-%m-%d")}"\n')
        out.write('author: "Neko Speech Team"\n')
        out.write("---\n\n")

        # 目录
        out.write("\\tableofcontents\n\n\\newpage\n\n")

        # 各章节
        for ch_file in chapters:
            content = ch_file.read_text()
            out.write(content)
            out.write("\n\n\\newpage\n\n")

    print(f"\n合并文件: {combined}")

    # 使用 pandoc 生成 PDF
    cmd = [
        "pandoc",
        str(combined),
        "-o", output_file,
        "--pdf-engine=xelatex",
        "-V", "mainfont=Noto Sans CJK SC",
        "-V", "monofont=Noto Sans Mono CJK SC",
        "-V", "geometry:margin=2.5cm",
        "-V", "fontsize=12pt",
        "--toc",
        "--toc-depth=2",
        "-N",
    ]

    print(f"\n生成 PDF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ PDF 生成成功: {output_file}")
        return True
    else:
        print(f"❌ PDF 生成失败:\n{result.stderr}")
        return False


def generate_html(output_file="neko-speech-textbook.html"):
    """生成 HTML 版本"""
    chapters = collect_chapters()
    if not chapters:
        return False

    combined = Path("build/combined.md")
    combined.parent.mkdir(exist_ok=True)

    with open(combined, "w") as out:
        out.write("---\n")
        out.write('title: "从 WaveNet 到 Omni：语音合成教科书"\n')
        out.write(f'date: "{datetime.now().strftime("%Y-%m-%d")}"\n')
        out.write("---\n\n")

        for ch_file in chapters:
            content = ch_file.read_text()
            out.write(content)
            out.write("\n\n---\n\n")

    cmd = [
        "pandoc",
        str(combined),
        "-o", output_file,
        "--standalone",
        "--toc",
        "--toc-depth=2",
        "--metadata", f"title=从 WaveNet 到 Omni：语音合成教科书",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ HTML 生成成功: {output_file}")
        return True
    else:
        print(f"❌ HTML 生成失败:\n{result.stderr}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成教科书 PDF/HTML")
    parser.add_argument("--format", choices=["pdf", "html", "both"], default="both")
    parser.add_argument("--output", type=str, default="build/neko-speech-textbook")
    args = parser.parse_args()

    if not check_pandoc():
        print("pandoc 未安装，正在安装...")
        install_pandoc()

    if args.format in ("pdf", "both"):
        generate_pdf(f"{args.output}.pdf")

    if args.format in ("html", "both"):
        generate_html(f"{args.output}.html")

    print("\n完成！")
