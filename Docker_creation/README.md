
# Docker creation on Jetson Nano

## Structure of Docker creation folder:

```
<Main Folder>/
├── app/
├── exporter.py
├── final_app.py
├── node_exporter 
├── Dockerfile
└── requirements.txt
```

**Note**: We need to have node exporter binary file in the folder

## Choice of Docker Image:
1. The idea was to make docker image as lighweight as possible due to computational constraints on the jetson.
2. We needed to run tegrastats to get GPU metrics, so it was better if we had base image for docker as:
   ```bash
     ARG JETPACK_VERSION=r35.1.0
     FROM nvcr.io/nvidia/l4t-base:${JETPACK_VERSION} as base
    ```
**Note**: Before running the below instructions, check the docker file, python files, requirements.txt thoroughly.



## On your workstation/Laptop (for quicker building)
1. Clone the github repo:
   ```bash
   ```
2. Go to directory where docker creation files are present
   ```bash
   cd Docker_creation
   ```
3. Building Docker image
    ```bash
    docker buildx build   --platform linux/arm64   --output type=docker   -t vehicle-detection-jetson:jetpack-4.6.2 .
    ```
4. Save the Docker image as tar file
    ```bash
    docker save -o jdocker.tar vehicle-detection-jetson:jetpack-4.6.2
    ```
5. Transfer it to jetson using scp or USB
   
## On Jetson Nano

1. Load the tar file
   ```bash
   docker load -i jdocker.tar
   ```
2. Run the docker image
   ```bash
   docker run -it --runtime nvidia --privileged --network host -v /home/jetson/App_project/stock_videos:/app/process_video -v /home/jetson/App_project/feedback/images:/app/feedback/images -v /home/jetson/App_project/feedback/labels:/app/feedback/labels -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro vehicle-detection-jetson:jetpack-4.6.2
   ```
3. For running next time
   ```bash
   docker start <container-name>
   ```
4. For stopping the container
   ```bash
   docker stop <conatiner-name>

