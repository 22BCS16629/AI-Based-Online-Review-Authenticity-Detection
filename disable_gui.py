import os

files = ['src/visualization.py', 'src/evaluation.py', 'src/statistical_analysis.py']

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Revert to headless mode to prevent GUI crashes on Windows
    content = content.replace("# matplotlib.use('Agg')", "matplotlib.use('Agg')")
    
    # Revert to close to avoid memory leaks and hanging processes
    content = content.replace("plt.show(block=False)", "plt.close()")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Reverted to Agg headless safe mode.")
