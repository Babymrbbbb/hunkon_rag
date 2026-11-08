"""多模态文本抽取 v1.0（纯标准库，零依赖）。

能力与边界（诚实）：
  - .md / .txt / .markdown：直接读取
  - .xlsx：zip + XML 解析 sharedStrings + 单元格 → Markdown 表格
  - .docx：zip + XML 解析段落 + 表格（表格转 Markdown）
  - .pdf ：文本流 + FlateDecode 压缩流解压（zlib），扫描件（纯图片）无法抽取

设计说明：
  本模块是生产知识库 kb_extract.py 的开源复刻（能力等价，实现独立）：
  - 不依赖 openpyxl / PyPDF2 等第三方库
  - 不硬编码任何绝对路径（参考实现版本依赖环境变量 KB_EXTRA_SOURCES 指定路径）
  - 统一入口 read_text(fp) -> str，失败返回 ""

用法：
  from app.extract import read_text
  md = read_text("门店数据.xlsx")   # -> Markdown 表格
"""
import os
import re
import zipfile
import zlib

TEXT_EXTS = {".md", ".txt", ".markdown"}


def _unescape_xml(s: str) -> str:
    """反转义 XML 实体。"""
    return (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


def _cell_text(c_xml: str, shared: list) -> str:
    """解析一个 <c> 单元格，返回文本。c_xml 为 <c ...>...</c> 原始 XML。"""
    v = re.search(r"<v>(.*?)</v>", c_xml, re.DOTALL)
    if v:
        val = v.group(1)
        # 共享字符串：t="s" 且值为索引
        if re.search(r't="s"', c_xml) and val.isdigit() and int(val) < len(shared):
            return shared[int(val)]
        return _unescape_xml(val)
    itext = re.findall(r"<t[^>]*>(.*?)</t>", c_xml, re.DOTALL)
    return _unescape_xml("".join(itext))


def _table_to_markdown(rows: list) -> str:
    """把二维表格转成 Markdown 表格。rows: list[list[str]]。"""
    md = []
    for i, row in enumerate(rows):
        cells = [c.strip().replace("|", "\\|") for c in row]
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("|" + "---|" * len(cells))
    return "\n".join(md)


def _extract_xlsx(fp: str) -> str:
    """xlsx = zip of XML。提取 sharedStrings + 各 worksheet 单元格 → Markdown。"""
    try:
        z = zipfile.ZipFile(fp)
    except Exception:
        return ""
    shared: list = []
    try:
        data = z.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
        for si in re.findall(r"<si>(.*?)</si>", data, re.DOTALL):
            texts = re.findall(r"<t[^>]*>(.*?)</t>", si, re.DOTALL)
            shared.append(_unescape_xml("".join(texts)))
    except KeyError:
        pass
    md_parts = []
    try:
        names = [n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]
        for name in sorted(names):
            xml = z.read(name).decode("utf-8", errors="ignore")
            rows = []
            for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.DOTALL):
                cells = []
                # 匹配整个 <c ...>...</c>，保留开标签属性（t="s" 判断需要）
                for cm in re.finditer(r"<c\b[^>]*>.*?</c>", row, re.DOTALL):
                    c_whole = cm.group(0)
                    cells.append(_cell_text(c_whole, shared))
                rows.append(cells)
            # 过滤全空行
            rows = [r for r in rows if any(x.strip() for x in r)]
            if rows:
                md_parts.append(_table_to_markdown(rows))
    except Exception:
        pass
    return "\n\n".join(md_parts)


def _extract_docx(fp: str) -> str:
    """docx = zip of XML。提取 word/document.xml 的段落 + 表格（转 Markdown）。"""
    try:
        z = zipfile.ZipFile(fp)
    except Exception:
        return ""
    try:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return ""

    out = []
    # 先提取所有表格块
    tbl_blocks = []
    for m in re.finditer(r"<w:tbl\b[^>]*>.*?</w:tbl>", xml, re.DOTALL):
        tbl_blocks.append((m.start(), m.end(), m.group(0)))

    # 从 XML 中剔除表格块，剩余部分提取段落（避免表格内单元格段落被重复提取）
    xml_no_tbl = xml
    for start, end, _blk in sorted(tbl_blocks, reverse=True):
        xml_no_tbl = xml_no_tbl[:start] + xml_no_tbl[end:]

    # 段落：从剔除表格后的 XML 提取
    for m in re.finditer(r"<w:p\b[^>]*>.*?</w:p>", xml_no_tbl, re.DOTALL):
        texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", m.group(0), re.DOTALL)
        text = _unescape_xml("".join(texts)).strip()
        if text:
            out.append(text)

    # 表格：按原顺序输出
    for _start, _end, blk in sorted(tbl_blocks):
        rows = []
        for tr in re.findall(r"<w:tr\b[^>]*>.*?</w:tr>", blk, re.DOTALL):
            cells = []
            for tc in re.findall(r"<w:tc\b[^>]*>.*?</w:tc>", tr, re.DOTALL):
                t = "".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", tc, re.DOTALL))
                cells.append(_unescape_xml(t).strip())
            rows.append(cells)
        rows = [r for r in rows if any(x for x in r)]
        if rows:
            out.append(_table_to_markdown(rows))
    return "\n\n".join(out)


def _parse_pdf_content(data: bytes) -> list:
    """从 PDF 内容流中抓 Tj / TJ 文本。"""
    texts = []

    def _decode(s: bytes) -> str:
        s = (
            s.replace(b"\\(", b"(")
            .replace(b"\\)", b")")
            .replace(b"\\n", b" ")
            .replace(b"\\r", b" ")
            .replace(b"\\t", b" ")
            .replace(b"\\\\", b"\\")
        )
        try:
            return s.decode("latin-1")
        except Exception:
            return ""

    # Tj: (text) Tj
    for m in re.finditer(rb"\((?:\\.|[^\\()])*\)\s*Tj", data):
        s = m.group(0)
        s = s[: s.rfind(b")") + 1]
        texts.append(_decode(s[1:-1]))
    # TJ: [(text) -120 (text)] TJ
    for m in re.finditer(rb"\[(.*?)\]\s*TJ", data, re.DOTALL):
        inner = m.group(1)
        parts = re.findall(rb"\((?:\\.|[^\\()])*\)", inner)
        for p in parts:
            texts.append(_decode(p[1:-1]))
    return [t for t in texts if t.strip()]


def _extract_pdf_text_from_stream(raw: bytes) -> str:
    """从 PDF 原始字节中提取所有 Tj/TJ 文本，支持普通流与 FlateDecode 流。"""
    texts: list = []

    # 1) 普通流：直接抓 ()Tj / []TJ
    for m in re.finditer(rb"\((?:\\.|[^\\()])*\)\s*Tj|\[.*?\]\s*TJ", raw, re.DOTALL):
        texts.extend(_parse_pdf_content(m.group(0)))
    # 2) 对象流解压：找 stream...endstream，对 FlateDecode 解压后抓文本
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL):
        data = m.group(1)
        if b"FlateDecode" in raw[max(0, m.start() - 200):m.start()]:
            try:
                dec = zlib.decompress(data)
            except Exception:
                try:
                    dec = zlib.decompress(data, -15)
                except Exception:
                    continue
            texts.extend(_parse_pdf_content(dec))
        else:
            texts.extend(_parse_pdf_content(data))

    return " ".join(t for t in texts if t.strip())


def _extract_pdf(fp: str) -> str:
    """尽力而为的 PDF 文本抽取。支持普通文本流 + FlateDecode 压缩流。"""
    try:
        raw = open(fp, "rb").read()
    except Exception:
        return ""
    return _extract_pdf_text_from_stream(raw)


def read_text(fp: str) -> str:
    """按扩展名统一抽取文本。失败返回 ""。"""
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext in TEXT_EXTS:
            return open(fp, "r", encoding="utf-8", errors="ignore").read()
        if ext == ".xlsx":
            return _extract_xlsx(fp)
        if ext == ".docx":
            return _extract_docx(fp)
        if ext == ".pdf":
            return _extract_pdf(fp)
    except Exception:
        return ""
    return ""