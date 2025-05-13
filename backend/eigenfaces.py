#!/usr/bin/env python
"""
Eigenfaces System - Command-line interface
"""
import argparse
import os
import sys
import importlib.util

def load_script(script_name):
    """Load a script from the faces/scripts directory"""
    script_path = os.path.join('faces', 'scripts', f"{script_name}.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found")
        return None
    
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def main():
    parser = argparse.ArgumentParser(description='Eigenfaces Facial Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register face command
    register_parser = subparsers.add_parser('register', help='Register a new face')
    register_parser.add_argument('name', help='Name of the person')
    register_parser.add_argument('image_path', help='Path to face image file')
    
    # Test recognition command
    test_parser = subparsers.add_parser('test', help='Test recognition with an image')
    test_parser.add_argument('image_path', help='Path to face image file')
    test_parser.add_argument('--threshold', type=float, default=12.0, 
                            help='Recognition threshold (default: 12.0)')
    
    # Retrain model command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain the model')
    retrain_parser.add_argument('--max-eigenfaces', type=int, default=40,
                              help='Maximum number of eigenfaces to compute (default: 40)')
    retrain_parser.add_argument('--batch-size', type=int, default=5,
                              help='Number of images to process in one batch (default: 5)')
    retrain_parser.add_argument('--memory-report', action='store_true',
                              help='Show memory usage during training')
    
    # Evaluate model command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model accuracy')
    
    args = parser.parse_args()
    
    if args.command == 'register':
        module = load_script('register_face')
        if module:
            module.register_new_face(args.name, args.image_path)
    
    elif args.command == 'test':
        module = load_script('test_own_face')
        if module:
            module.recognize_face(args.image_path, args.threshold)
    
    elif args.command == 'retrain':
        module = load_script('retrain_model')
        if module:
            module.retrain_model(args.max_eigenfaces, args.batch_size, args.memory_report)
    
    elif args.command == 'evaluate':
        module = load_script('test_recognition')
        if module:
            module.test_recognition()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 