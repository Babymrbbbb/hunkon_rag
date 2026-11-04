"""文档解析模块测试：md/xlsx/docx/pdf 四种格式都能抽取文本。

背景：
  生产知识库有 kb_extract.py（2026-11-03 v1.1），hunkon_rag 移植为 app/extract.py。
  这些测试用纯标准库 zipfile 手工构造最小合法 .docx/.xlsx，避免引入第三方依赖。
  PDF 用文本流（非压缩）构造，验证 Tj/TJ 提取。
"""
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extract import read_text  # noqa: E402


def _write_tmp(suffix: str, content: bytes) -> str:
    """写临时文件，返回路径。"""
    p = Path(tempfile.mkdtemp(prefix="hunkon_rag_extract_")) / f"test{suffix}"
    p.write_bytes(content)
    return str(p)


# ---------- xlsx 构造 ----------

def _make_xlsx() -> bytes:
    """构造最小合法 xlsx：sharedStrings 含 2 个字符串，sheet1 引用它们。"""
    buf = []
    # [Content_Types].xml
    buf.append('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
               '<Default Extension="xml" ContentType="application/xml"/>'
               '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
               '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
               '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
               '</Types>')
    # workbook.xml
    buf.append('<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets>'
               '</workbook>')
    # sharedStrings.xml
    buf.append('<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">'
               '<si><t>产品</t></si><si><t>冰面包（非在售）</t></si></sst>')
    # sheet1.xml：A1=共享串0(产品) B1=共享串1(冰面包) A2=数字 47
    buf.append('<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<sheetData>'
               '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
               '<row r="2"><c r="A2"><v>47</v></c></row>'
               '</sheetData></worksheet>')
    # workbook.xml.rels
    buf.append('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
               '</Relationships>')

    f = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    f.close()
    with zipfile.ZipFile(f.name, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", buf[0])
        z.writestr("xl/workbook.xml", buf[1])
        z.writestr("xl/sharedStrings.xml", buf[2])
        z.writestr("xl/worksheets/sheet1.xml", buf[3])
        z.writestr("xl/_rels/workbook.xml.rels", buf[4])
    data = Path(f.name).read_bytes()
    Path(f.name).unlink()
    return data


# ---------- docx 构造 ----------

def _make_docx() -> bytes:
    """构造最小合法 docx：一个段落 + 一个 2x2 表格。"""
    document = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>冰面包非在售（调研对象）</w:t></w:r></w:p>'
        '<w:tbl>'
        '<w:tr><w:tc><w:p><w:r><w:t>城市</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>门店数</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:p><w:r><w:t>清远</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>40</w:t></w:r></w:p></w:tc></w:tr>'
        '</w:tbl>'
        '</w:body></w:document>'
    )
    f = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    f.close()
    with zipfile.ZipFile(f.name, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", document)
    data = Path(f.name).read_bytes()
    Path(f.name).unlink()
    return data


# ---------- pdf 构造 ----------

def _make_pdf() -> bytes:
    """构造最小 PDF：一页含 (Hello) Tj 和 (Hunkon) Tj。"""
    content = b"BT /F1 12 Tf 72 720 Td (Hello) Tj 0 -14 Td (Hunkon) Tj ET"
    objects = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"trailer<</Root 1 0 R/Size 6>>\n%%EOF\n"
    )
    return objects


# ---------- 测试 ----------

def test_read_markdown():
    p = _write_tmp(".md", "# 标题\n正文内容".encode("utf-8"))
    assert "标题" in read_text(p)
    assert "正文内容" in read_text(p)


def test_read_txt():
    p = _write_tmp(".txt", "纯文本内容".encode("utf-8"))
    assert read_text(p) == "纯文本内容"


def test_read_xlsx():
    p = _write_tmp(".xlsx", _make_xlsx())
    md = read_text(p)
    assert "产品" in md, f"共享字符串未提取: {md}"
    assert "冰面包（非在售）" in md, f"共享字符串未提取: {md}"
    assert "47" in md, f"数字单元格未提取: {md}"
    # 应转成 Markdown 表格
    assert "---" in md, f"未生成表头分隔: {md}"


def test_read_docx():
    p = _write_tmp(".docx", _make_docx())
    md = read_text(p)
    assert "冰面包非在售" in md, f"段落未提取: {md}"
    assert "城市" in md and "门店数" in md, f"表格头未提取: {md}"
    assert "清远" in md and "40" in md, f"表格行未提取: {md}"
    assert "---" in md, f"未生成表头分隔: {md}"


def test_read_pdf_text_stream():
    p = _write_tmp(".pdf", _make_pdf())
    txt = read_text(p)
    assert "Hello" in txt, f"PDF 文本流未提取: {txt}"
    assert "Hunkon" in txt, f"PDF 文本流未提取: {txt}"


def test_read_pdf_flate_stream():
    """FlateDecode 压缩流应能被 zlib 解压并提取文本。"""
    content = b"BT (Compressed) Tj ET"
    compressed = zlib.compress(content)
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(compressed)).encode() + b"/Filter/FlateDecode>>stream\n" + compressed + b"\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"trailer<</Root 1 0 R/Size 6>>\n%%EOF\n"
    )
    p = _write_tmp(".pdf", pdf)
    txt = read_text(p)
    assert "Compressed" in txt, f"FlateDecode 流未提取: {txt}"


def test_unsupported_extension_returns_empty():
    p = _write_tmp(".xyz", b"whatever")
    assert read_text(p) == ""


def test_missing_file_returns_empty():
    assert read_text(str(Path(tempfile.mkdtemp()) / "nope.md")) == ""