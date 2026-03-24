import psutil

def get_system_usage():
    cpu = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory().percent
    return cpu, memory

def get_resource_score():
    cpu, memory = get_system_usage()
    return (cpu / 100 + memory / 100) / 2