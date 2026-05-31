from pathlib import Path
import html
import re

src = Path('/Users/wendy/svg-demo/paper_update/updated_paper_full.md')
out = Path('/Users/wendy/svg-demo/paper_update/updated_paper_full.html')
text = src.read_text(encoding='utf-8')

lines = text.splitlines()
html_lines = []
in_table = False
in_list = False

def close_blocks():
    global in_table, in_list
    if in_table:
        html_lines.append('</table>')
        in_table = False
    if in_list:
        html_lines.append('</ul>')
        in_list = False

for line in lines:
    s = line.rstrip('\n')
    if not s.strip():
        close_blocks()
        continue

    if s.startswith('# '):
        close_blocks(); html_lines.append(f"<h1>{html.escape(s[2:])}</h1>"); continue
    if s.startswith('## '):
        close_blocks(); html_lines.append(f"<h2>{html.escape(s[3:])}</h2>"); continue
    if s.startswith('### '):
        close_blocks(); html_lines.append(f"<h3>{html.escape(s[4:])}</h3>"); continue

    if s.startswith('|') and s.endswith('|'):
        cells = [c.strip() for c in s.strip('|').split('|')]
        if len(cells) > 1 and all(set(c.replace(':','').replace('-','').strip()) == set() for c in cells):
            continue
        if not in_table:
            close_blocks()
            html_lines.append('<table>')
            in_table = True
            html_lines.append('<tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cells) + '</tr>')
        else:
            html_lines.append('<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in cells) + '</tr>')
        continue

    if s.strip().startswith('- '):
        if not in_list:
            close_blocks(); html_lines.append('<ul>'); in_list = True
        html_lines.append(f"<li>{html.escape(s.strip()[2:])}</li>")
        continue

    m = re.match(r'!\[(.*?)\]\((.*?)\)', s.strip())
    if m:
        close_blocks()
        alt = html.escape(m.group(1))
        srcp = html.escape(m.group(2))
        html_lines.append(f'<figure><img src="{srcp}" alt="{alt}"><figcaption>{alt}</figcaption></figure>')
        continue

    if s.strip().startswith('---'):
        close_blocks(); html_lines.append('<hr>'); continue

    # inline code
    p = html.escape(s)
    p = re.sub(r'`([^`]+)`', r'<code>\1</code>', p)
    close_blocks()
    html_lines.append(f'<p>{p}</p>')

close_blocks()

page = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>论文正文</title>
<style>
body {{ font-family: "Songti SC", "STSong", serif; max-width: 860px; margin: 24px auto; line-height: 1.8; color: #111; }}
h1, h2, h3 {{ font-family: "Heiti SC", "PingFang SC", sans-serif; }}
h1 {{ text-align: center; font-size: 30px; margin-bottom: 10px; }}
h2 {{ margin-top: 28px; border-left: 4px solid #333; padding-left: 10px; }}
h3 {{ margin-top: 18px; }}
p {{ text-indent: 2em; margin: 8px 0; }}
figure {{ margin: 16px auto; text-align: center; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
figcaption {{ font-size: 14px; color: #444; margin-top: 6px; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }}
th, td {{ border: 1px solid #777; padding: 6px 8px; text-align: center; }}
th {{ background: #f3f3f3; }}
ul {{ margin: 8px 0 8px 2em; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
hr {{ border: none; border-top: 1px solid #ddd; margin: 18px 0; }}
</style>
</head>
<body>
{''.join(html_lines)}
</body>
</html>'''

out.write_text(page, encoding='utf-8')
print(out)
