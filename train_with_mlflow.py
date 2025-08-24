import mlflow
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
import yaml
import json
import hashlib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params_v1.yaml", help="Path to params file")
args = parser.parse_args()
yolov5_path = "app/core/models/yolov5/train.py"

with open(args.params) as f:
    config = yaml.safe_load(f)
params = config['train']  

def prepare_retraining(params):
    """Update data config for new classes"""
    with open(params["data"]) as f:
        data_cfg = yaml.safe_load(f)
    
    # Load drift report
    with open("monitor/drift_report.json") as f:
        drift = json.load(f)
    
    if drift["needs_retrain"]:
        # print(f"Adding new classes: {drift['new_classes']}")
        # data_cfg["names"].extend(drift["new_classes"])
        # data_cfg["nc"] = len(data_cfg["names"])

        # Save updated config
        with open(params["data"], "w") as f:
            yaml.dump(data_cfg, f)
        
        # Update run name
        return f"retrain_{datetime.now().strftime('%Y%m%d')}"
    return params["name"]

# Modify the MLflow run section
run_name = prepare_retraining(params)

def get_data_version(data_path):
    dvc_file = data_path + '.dvc'
    if os.path.exists(dvc_file):
        with open(dvc_file, 'r') as f:
            for line in f:
                if 'md5:' in line:
                    return line.strip().split('md5:')[-1]
    return "unknown"

data_version = get_data_version(params["data"].split('/')[1])

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment('Vehicle_detection')  # or Plate_detection

# Start MLflow run
with mlflow.start_run(run_name=run_name):
   
    # Log hyperparameters
    mlflow.log_params(params)
    mlflow.log_param("dataset_version",data_version)
    mlflow.log_param("is_retrain", "retrain" in run_name)
    
    # Run YOLOv5 training
    cmd = f"""
    python {yolov5_path} \
        --img {params['img']} \
        --batch {params['batch']} \
        --epochs {params['epochs']} \
        --data {params['data']} \
        --weights {params['weights']} \
        --workers {params['workers']} \
        --project {params['project']} \
        --name {params['name']} \
        --exist-ok
    """
    subprocess.run(cmd, shell=True, check=True)
    
    # Log results (metrics + artifacts)
    exp_dir = Path(params["project"]) / params["name"]
    mlflow.log_artifacts(exp_dir)  # Saves all training outputs
    
    # Log specific metrics from results.csv
    results_csv = exp_dir / "results.csv"
    if results_csv.exists():
        mlflow.log_artifact(results_csv)
        
        # Read CSV and extract final epoch metrics
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Clean column names

        metric_names = [
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5_0.95',
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss'
        ]

        for _, row in df.iterrows():
            epoch = int(row['epoch']) if 'epoch' in row else int(row['               epoch'])
            for metric in metric_names:
                metric=metric.replace(":", "_")
                if metric in row:
                    mlflow.log_metric(metric, row[metric], step=epoch)

    # Cleanup drift report after successful retraining
    if "retrain" in run_name and os.path.exists("monitor/drift_report.json"):
        os.remove("monitor/drift_report.json")