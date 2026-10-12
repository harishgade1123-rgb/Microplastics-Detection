import os
import sys

# Add project root directory to sys.path so server and engine modules can be resolved
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Expose WSGI application for Vercel Serverless Functions
from server import app
