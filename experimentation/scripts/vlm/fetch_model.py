import os, sys
from huggingface_hub import snapshot_download
tok = [l.split("=", 1)[1].strip().strip("\"'") for l in open("/home/colligo/DatasetDiscovery/.env") if l.startswith("HUGGINGFACE_TOKEN=")][0]
p = snapshot_download(sys.argv[1], token=tok, max_workers=8)
print("model at", p)
