"""
Command Line Interface for XSeas.

This module provides a CLI for managing XSeas projects and workflows.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from xseas.manager import SeasonDetect, create_sample_config, create_directory_structure


def create_project(args) -> None:
    """Create a new XSeas project."""
    project_path = Path(args.path)
    
    if project_path.exists():
        print(f"📁 Directory already exists: {project_path}")
        
        # Check if it's already an XSeas project
        config_path = project_path / 'config' / 'config.yaml'
        if config_path.exists():
            print("⚠️  This appears to be an existing XSeas project!")
            response = input("Do you want to continue and potentially overwrite the config? (y/N): ")
            if response.lower() != 'y':
                print("❌ Project creation cancelled")
                return
        
        print("🔧 Setting up XSeas structure in existing directory...")
    else:
        print(f"🚀 Creating new XSeas project at: {project_path}")
    
    # Create directory structure (will skip existing directories)
    create_directory_structure(project_path)
    
    # Handle configuration file
    config_path = project_path / 'config' / 'config.yaml'
    if config_path.exists():
        print(f"📝 Configuration file already exists: {config_path}")
        response = input("Do you want to create a backup and generate a new config? (y/N): ")
        if response.lower() == 'y':
            # Create backup
            backup_path = config_path.with_suffix('.yaml.backup')
            config_path.rename(backup_path)
            print(f"💾 Backup created: {backup_path}")
            create_sample_config(config_path)
        else:
            print("✅ Keeping existing configuration file")
    else:
        create_sample_config(config_path)
    
    print("✅ Project setup completed!")
    print(f"📝 Configuration file: {config_path}")
    print(f"📂 Add your data to: {project_path / 'data'}")


def run_workflow(args) -> None:
    """Run XSeas workflow."""
    try:
        detector = SeasonDetect(args.path, args.config)
        
        if args.step == 'all':
            detector.run_full_workflow()
        elif args.step == 'check':
            detector.check_data_availability()
        elif args.step == 'prenorm':
            detector.prenormalize_ERA5()
            detector.prenormalize_CMIP6()
        elif args.step == 'cluster':
            detector.perform_clustering()
        elif args.step == 'train':
            detector.train_perceptron_models()
        else:
            print(f"❌ Unknown step: {args.step}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def show_status(args) -> None:
    """Show project status."""
    try:
        detector = SeasonDetect(args.path, args.config)
        print(detector)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="XSeas - Xarray-based tools for meteorological Seasons detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xseas create ./my_project                    # Create new project
  xseas status ./my_project                    # Show project status
  xseas run ./my_project --step all           # Run full workflow
  xseas run ./my_project --step cluster       # Run only clustering
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser(
        'create', 
        help='Create a new XSeas project'
    )
    create_parser.add_argument(
        'path', 
        help='Path where to create the project'
    )
    create_parser.set_defaults(func=create_project)
    
    # Status command
    status_parser = subparsers.add_parser(
        'status', 
        help='Show project status'
    )
    status_parser.add_argument(
        'path', 
        help='Path to XSeas project'
    )
    status_parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file name (default: config.yaml)'
    )
    status_parser.set_defaults(func=show_status)
    
    # Run command
    run_parser = subparsers.add_parser(
        'run', 
        help='Run XSeas workflow'
    )
    run_parser.add_argument(
        'path', 
        help='Path to XSeas project'
    )
    run_parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file name (default: config.yaml)'
    )
    run_parser.add_argument(
        '--step', 
        choices=['all', 'check', 'prenorm', 'cluster', 'train'],
        default='all',
        help='Workflow step to run (default: all)'
    )
    run_parser.set_defaults(func=run_workflow)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
