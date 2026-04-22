#!/usr/bin/env python3
"""
Easy runner script for federated learning
"""
import os
import sys
import argparse
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Federated Learning for Fraud Detection')
    parser.add_argument('--rounds', type=int, default=20, help='Number of training rounds')
    parser.add_argument('--privacy', action='store_true', help='Enable differential privacy')
    parser.add_argument('--quick', action='store_true', help='Quick test with small samples')
    parser.add_argument('--save', action='store_true', help='Save model and results')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     FEDERATED LEARNING FOR FRAUD DETECTION                ║
    ║     Bank Account Fraud + Credit Card Fraud                ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📊 Configuration:")
    print(f"   • Rounds: {args.rounds}")
    print(f"   • Privacy: {'YES' if args.privacy else 'NO'}")
    print(f"   • Quick test: {'YES' if args.quick else 'NO'}")
    print(f"   • Save results: {'YES' if args.save else 'NO'}")
    print()
    
    # Build command
    cmd = f"python3 main.py --rounds {args.rounds}"
    if args.privacy:
        cmd += " --privacy"
    if args.save:
        cmd += " --save_model"
    
    # Run
    os.system(cmd)
