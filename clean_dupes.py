import re

with open('master.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Split into paragraphs based on double newlines (or more)
paragraphs = re.split(r'\n{2,}', content)

new_paragraphs = []
i = 0
while i < len(paragraphs):
    p = paragraphs[i].strip()
    
    # Check if this paragraph and the next one are identical to the two after them
    # Pattern: A B A B
    if i + 3 < len(paragraphs):
        p1 = paragraphs[i].strip()
        p2 = paragraphs[i+1].strip()
        p3 = paragraphs[i+2].strip()
        p4 = paragraphs[i+3].strip()
        
        if p1 == p3 and p2 == p4 and len(p1) > 20 and len(p2) > 20:
            # We found a duplicated block of two paragraphs!
            new_paragraphs.append(p1)
            new_paragraphs.append(p2)
            i += 4  # Skip the duplicates
            continue
            
    # Check if just this paragraph is duplicated (A A)
    if i + 1 < len(paragraphs):
        p1 = paragraphs[i].strip()
        p2 = paragraphs[i+1].strip()
        if p1 == p2 and len(p1) > 20:
            new_paragraphs.append(p1)
            i += 2
            continue

    new_paragraphs.append(paragraphs[i])
    i += 1

# Join back with double newlines
cleaned_content = "\n\n".join(new_paragraphs)

with open('master.tex', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("Duplicates removed.")
