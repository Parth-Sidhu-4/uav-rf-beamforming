import re
with open(r'master.tex', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    s = line.strip()
    if re.search(r'\\(sub)*section\{|\\paragraph\{', s):
        print(f'{i:5d}: {s[:120]}')
