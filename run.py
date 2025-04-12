import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture its output"""
    print(f"\nRunning {script_name}...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myenv', 'Scripts', 'python.exe')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    result = subprocess.run([python_exe, script_path], capture_output=True, text=True, encoding='utf-8', env=env)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        exit(1)
    print(f"{script_name} completed successfully.")

def save_figure(name):
    """Save the current figure with a specific name"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Saved_charts')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Saved {name} chart as {filepath}")
    plt.close()

def custom_show():
    """Custom show function that saves the current figure"""
    current_fig = plt.gcf()
    fig_num = current_fig.number
    chart_name = f'chart_{fig_num}'
    save_figure(chart_name)

# Create the Saved_charts directory
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Saved_charts')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Run gpay.py and paytm.py
run_script('gpay.py')
run_script('paytm.py')

# Replace plt.show with our custom function
original_show = plt.show
plt.show = custom_show

# Run main.py with modified show function
run_script('main.py')

# Restore original show function
plt.show = original_show

print("\nAll scripts completed successfully!")
print(f"Charts have been automatically saved in the '{save_dir}' directory.")