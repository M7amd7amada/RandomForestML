# app/__init__.py

from flask import Flask

app = Flask(__name__)

# Import routes at the end to avoid circular import issues
from app import routes
