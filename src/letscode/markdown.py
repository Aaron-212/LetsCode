import html2text


def html_to_markdown(content: str) -> str:
    """Convert LeetCode's rich HTML problem content to terminal-friendly markdown."""
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_links = False
    return converter.handle(content).strip()
