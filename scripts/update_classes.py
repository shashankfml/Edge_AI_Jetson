#!/usr/bin/env python3
import yaml
from pathlib import Path
import json

CORE_DIR = Path(__file__).parent.parent.resolve()
feedback_dir = CORE_DIR / "feedback" / "labels"

def update_classes():
    data_yaml = CORE_DIR / "data" / "vehicle_data_v1" / "vehicle_data.yaml"
    
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    current_classes = data['names']
    #initial_count = len(current_classes)
    
    # Use a set to automatically handle duplicates
    new_classes = set()
    
    for label_file in feedback_dir.glob("*.txt"):
        # More robust class name extraction
        parts = label_file.stem.split('_')
        cls_name = parts[0] if parts else None
        
        if cls_name and cls_name not in current_classes:
            new_classes.add(cls_name)
    
    if new_classes:
        # Add new classes and deduplicate
        updated_classes = current_classes + list(new_classes)
        data['names'] = updated_classes #sorted(list(set(updated_classes)))  # Remove duplicates
        data['nc'] = len(updated_classes)
        
        with open(data_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)  # sort_keys=False maintains order
        
        print(f"Added new classes: {new_classes}")
    else:
        print("No new classes found in feedback data")

if __name__ == "__main__":
    update_classes()