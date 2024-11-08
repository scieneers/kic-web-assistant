import multiprocessing

max_requests = 1000
max_requests_jitter = 50
log_file = "-"
bind = "0.0.0.0"

# https://learn.microsoft.com/en-us/troubleshoot/azure/app-service/web-apps-performance-faqs#why-does-my-request-time-out-after-230-seconds
timeout = 230

num_cpus = multiprocessing.cpu_count()
workers = min((num_cpus * 2) + 1, 8)
worker_class = "uvicorn.workers.UvicornWorker"
