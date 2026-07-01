import re

with open('master.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Find the EDA section
eda_start = tex.find(r"\section{Exploratory Data Analysis}")
outcomes_start = tex.find(r"%  4. OUTCOMES / DISCUSSION")

eda_tex = tex[eda_start:outcomes_start]

# Regex to find figures and their following text
pattern = re.compile(
    r"\\begin\{figure\}.*?"
    r"\\includegraphics\[.*?\]\{([^}]+)\}.*?"
    r"\\caption\{([^}]+)\}.*?"
    r"\\label\{([^}]+)\}.*?"
    r"\\end\{figure\}\s*"
    r"([^\\]+)",  # Capture the following paragraph (stops at next latex command)
    re.DOTALL
)

matches = pattern.findall(eda_tex)

for i, (img, caption, label, desc) in enumerate(matches):
    print(f"--- FIGURE {i+1} ---")
    print(f"IMAGE:   {img}")
    print(f"CAPTION: {caption.strip()}")
    print(f"LABEL:   {label}")
    # Print the first sentence or two of the description
    desc_clean = " ".join(desc.strip().split())
    print(f"DESC:    {desc_clean[:150]}...")
    print()
