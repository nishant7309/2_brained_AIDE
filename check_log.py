
import json
import os

log_dir = r"c:\Users\opnin\OneDrive\Desktop\aide\aideml\logs\2-funny-acrid-chamois"
journal_path = os.path.join(log_dir, "journal.json")

try:
    with open(journal_path, "r") as f:
        data = json.load(f)
        
    for i, node in enumerate(data.get('nodes', [])):
        print(f"Node {i}: Buggy={node.get('is_buggy')}")
        print(f"Exc type: {node.get('exc_type')}")
        print(f"Exc info: {node.get('exc_info')}")
        print(f"Execution time: {node.get('exec_time')}")
except Exception as e:
    print(f"Error reading journal: {e}")
