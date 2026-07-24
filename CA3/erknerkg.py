from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

# Read your Python file
with open('CA3-PART3.py', 'r') as f:
    code = f.read()

# Generate highlighted HTML
lexer = PythonLexer()
formatter = HtmlFormatter(full=True, style='friendly')
html = highlight(code, lexer, formatter)

# Save to an HTML file
with open('myfile.html', 'w') as f:
    f.write(html)