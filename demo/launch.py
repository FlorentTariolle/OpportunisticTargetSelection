#!/usr/bin/env python3
"""Launcher script for the adversarial attack demonstrator."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.demo import launch_demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch the Adversarial Attack Demonstrator")
    parser.add_argument("--share", action="store_true", help="Create a public link for sharing")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    
    args = parser.parse_args()
    
    print("Launching Adversarial Attack Demonstrator...")
    print(f"Server will be available at http://{args.host}:{args.port}")
    if args.share:
        print("Public sharing link will be generated")
    
    launch_demo(share=args.share, server_name=args.host, server_port=args.port)
