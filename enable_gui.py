import os

files = ['src/visualization.py', 'src/evaluation.py', 'src/statistical_analysis.py']

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Disable headless mode
    content = content.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')")
    
    # Don't close immediately so windows can stay open
    content = content.replace("plt.close()", "plt.show(block=False)")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Updated plot generation scripts.")
