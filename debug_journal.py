
import json
import os

log_dir = r"c:\Users\opnin\OneDrive\Desktop\aide\aideml\logs\2-laughing-aboriginal-lynx"
journal_path = os.path.join(log_dir, "journal.json")

try:
    with open(journal_path, "r") as f:
        data = json.load(f)
        
    for i, node in enumerate(data.get('nodes', [])):
        if node.get('is_buggy'):
            print(f"Node {i} is BUGGY")
            print(f"Exc type: {node.get('exc_type')}")
            print(f"Exc info: {node.get('exc_info')}")
            output = node.get('term_out', [])
            if output:
                print("Traceback/Output:")
                # Join the last 50 lines to see the error
                print("\n".join(output[-50:]))
except Exception as e:
    print(f"Error reading journal: {e}")
