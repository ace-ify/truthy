# Single worker: each worker loads its own ~2GB model copy; more workers => OOM/502 (see commit 5be1c88).
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
