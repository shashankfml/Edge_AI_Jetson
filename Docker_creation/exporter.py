import subprocess
import time
import re
from prometheus_client import start_http_server, Gauge
#import psutil

# Define Prometheus metrics
#cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percent')
#memory_usage = Gauge('system_memory_usage_percent', 'System memory usage percent')
gpu_usage = Gauge('jetson_gpu_usage_percent', 'GPU usage percent')
gpu_power = Gauge('jetson_gpu_power_mw', 'GPU power consumption in mW')

#def collect_system_metrics():
    #""" Collect CPU and memory usage metrics """
    #cpu_usage.set(psutil.cpu_percent())
    #memory_usage.set(psutil.virtual_memory().percent)

def collect_gpu_metrics():
    """ Collect GPU usage and power using tegrastats """
    try:
        tegrastats_proc = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        for output in tegrastats_proc.stdout:


        # Sample tegrastats line (may vary by JetPack version):
        # RAM 2013/3964MB (lfb 32x4MB) SWAP 0/1982MB (cached 0MB) CPU [0%@345,off,0%@345,0%@345] EMC_FREQ 0% GR3D_FREQ 20%@153 GPU_PWR 1234mW
            gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', output)
            power_match = re.search(r'GPU_PWR (\d+)mW', output)

            if gpu_match:
                gpu_usage.set(int(gpu_match.group(1)))
            if power_match:
                gpu_power.set(int(power_match.group(1)))

    except subprocess.CalledProcessError:
        print("Failed to run tegrastats")

def run_node_exporter():
    """ Run Node Exporter from a custom directory """
    #node_exporter_path = "/home/jetson/zzzjetson-metrics-exporter/node_exporter"
    node_exporter_path = "/usr/local/bin/node_exporter"  # Update this path as needed
    subprocess.Popen([node_exporter_path, "--web.listen-address=0.0.0.0:9100"])

def start_exporter():
    """ Start the Prometheus HTTP server for custom metrics """
    start_http_server(8000)
    print("Prometheus metrics server started on port 8000.")

if __name__ == '__main__':
    run_node_exporter()
    start_exporter()

    while True:
        #collect_system_metrics()
        collect_gpu_metrics()
        time.sleep(1)






