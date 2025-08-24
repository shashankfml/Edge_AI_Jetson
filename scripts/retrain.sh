#!/bin/bash
# Merge feedback data
# cp feedback/images/* data/vehicle_data_v1/train/images/
# cp feedback/labels/* data/vehicle_data_v1/train/labels/

# Update data.yaml
python scripts/update_classes.py

# Retrain using existing pipeline
python train_with_mlflow.py --params params_retrain.yaml

# Cleanup feedback (optional)
#rm -rf feedback/*