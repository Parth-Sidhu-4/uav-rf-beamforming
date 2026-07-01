with open('master.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].strip()
    if len(line) > 50 and line.startswith("Figure \\ref{fig:"):
        # check next 10 lines for the exact same line
        for j in range(i+1, min(i+15, len(lines))):
            if lines[j].strip() == line:
                print(f"DUPLICATE FOUND: lines {i+1} and {j+1}")
