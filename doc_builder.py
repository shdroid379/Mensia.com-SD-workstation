import io
import markdown
from xhtml2pdf import pisa
from docx import Document
from htmldocx import HtmlToDocx

def markdown_to_html(md_text: str) -> str:
    # Converts Markdown to HTML, ensuring tables and code blocks are parsed cleanly
    return markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

def build_pdf_buffer(md_text: str) -> io.BytesIO:
    html_content = markdown_to_html(md_text)
    
    # CSS styling injected specifically for xhtml2pdf parsing
    styled_html = f"""
    <html>
    <head>
        <style>
            @page {{ margin: 2cm; }}
            body {{ font-family: Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }}
            h1 {{ color: #000000; font-size: 18pt; border-bottom: 1px solid #000; padding-bottom: 4px; }}
            h2 {{ color: #333333; font-size: 14pt; margin-top: 15px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #dddddd; padding: 6px; text-align: left; }}
            th {{ background-color: #f4f4f4; font-weight: bold; }}
            code {{ background-color: #f0f0f0; padding: 2px 4px; font-family: Courier, monospace; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(styled_html, dest=buffer) # Renders direct to memory
    buffer.seek(0)
    
    if pisa_status.err:
        raise Exception("PDF Compilation Failed")
    return buffer

def build_docx_buffer(md_text: str) -> io.BytesIO:
    html_content = markdown_to_html(md_text)
    
    document = Document()
    new_parser = HtmlToDocx()
    # Converts the HTML elements (h1, p, table) into native Word docx elements
    new_parser.add_html_to_document(html_content, document)
    
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer

def build_markdown_buffer(md_text: str) -> io.BytesIO:
    # For researchers who just want the raw Markdown file for Notion/Obsidian
    buffer = io.BytesIO()
    buffer.write(md_text.encode('utf-8'))
    buffer.seek(0)
    return buffer