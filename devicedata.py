import pynvml as nvml
import psutil
import asyncio

time_interval = 0.1

def measure_current_GPU_memory_per_handle(handle, data_list):
    
    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
    data_list.append((memory.total - memory.free) / 1024**2)

def measure_current_GPU_usage_per_handle(handle, data_list):
    
    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
    data_list.append(utilization.gpu / 100)

def measure_current_CPU_usage_percent_per_pid(process: psutil.Process, data_list):
    
    CPU_util = process.cpu_percent(interval=0.1)
    data_list.append(CPU_util)


def measure_current_RAM_usage_per_pid(process: psutil.Process, data_list):
    
    current_RAM = process.memory_percent()
    data_list.append(current_RAM)