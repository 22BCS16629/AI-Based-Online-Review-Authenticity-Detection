"""Fix Unicode characters in all Python source files for Windows cp1252 compatibility."""
import os
import glob

replacements = {
    '\u2192': '->',   # right arrow
    '\u2500': '-',    # box drawing horizontal
    '\u2014': '-',    # em dash
    '\u00b1': '+/-',  # plus-minus
    '\u2588': '#',    # full block
}

files = glob.glob('src/**/*.py', recursive=True) + ['main.py', 'config.py']

for filepath in files:
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for old, new in replacements.items():
        content = content.replace(old, new)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
    else:
        print(f"OK:    {filepath}")

print("\nDone!")
