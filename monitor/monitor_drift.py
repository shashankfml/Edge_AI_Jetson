import json
from pathlib import Path
from collections import Counter

class DriftMonitor:
    def __init__(self, feedback_dir="feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.known_classes = {"Ambulance", "Bus", "Car", "Motorcycle", "Truck"}
        
    def check_new_classes(self):
        class_counts = Counter()
        for label_file in (self.feedback_dir/"labels").glob("*.txt"):
            with open(label_file) as f:
                parts = label_file.stem.split('_')
                class_name = parts[0] if parts else None
                #class_name=f.name.split('/')[-1].split('_')[0]
                class_id = f.read().split()[0]
                class_counts[class_name] += 1
        
        new_classes = {
            cls: count for cls, count in class_counts.items() 
            if cls not in self.known_classes and count >= 3
        }
        return new_classes

if __name__ == "__main__":
    monitor = DriftMonitor()
    new_classes = monitor.check_new_classes()
    
    with open("monitor/drift_report.json", "w") as f:
        json.dump({
            "new_classes": list(new_classes.keys()),
            "counts": dict(new_classes),
            "needs_retrain": len(new_classes) > 0
        }, f)
