"""The examples directory (its model-building scripts) is importable from every test."""
import os, sys
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
if EX not in sys.path:
    sys.path.insert(0, EX)
