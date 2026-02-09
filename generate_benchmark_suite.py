#!/usr/bin/env python3
"""
Fully automatic OCL constraint benchmark generation with advanced features.

Usage:
    python generate_benchmark_suite.py --config examples/example_suite.yaml
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress TensorFlow CPU feature info logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from modules.generation.benchmark.suite_config import BenchmarkSuite
from modules.generation.benchmark.suite_controller_enhanced import EnhancedSuiteController


def main():
    parser = argparse.ArgumentParser(
        description="Generate OCL constraint benchmark suite from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from example config
  python generate_benchmark_suite.py --config examples/example_suite.yaml
  
  # Silent mode
  python generate_benchmark_suite.py --config suite.yaml --quiet
  
  # Show version
  python generate_benchmark_suite.py --version
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=str,
        help='Path to suite configuration YAML file'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output (minimal logging)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (default)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging (very detailed)'
    )
    
    parser.add_argument(
        '--no-research-features',
        action='store_true',
        help='Disable all 6 advanced features (metadata, UNSAT, similarity, etc.)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='OCL Benchmark Generator v2.0 (Enhanced with Advanced Features)'
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("OCL Benchmark Suite Generator - Enhanced Edition")
    print("Automatic Benchmark Factory with Advanced Features")
    print(f"{'='*70}\n")
    print(f"Config: {args.config}")
    
    # Set verbosity
    verbose = not args.quiet if not args.verbose else True
    debug = args.debug
    enable_research = not args.no_research_features
    
    if debug:
        print("Debug logging ENABLED")
    if not enable_research:
        print("Advanced features DISABLED")
    
    try:
        # Load suite configuration
        suite = BenchmarkSuite.from_yaml(str(config_path))
        print(f"Loaded suite: {suite.suite_name} v{suite.version}")
        print(f"   Models: {len(suite.models)}")
        total_profiles = sum(len(m.profiles) for m in suite.models)
        print(f"   Profiles: {total_profiles}")
        print(f"   Output: {suite.output_root}")
        print(f"   Advanced Features: {'ENABLED' if enable_research else 'DISABLED'}")
        print()
        
        # Run generation with enhanced controller
        controller = EnhancedSuiteController(
            suite,
            verbose=verbose,
            debug=debug,
            enable_research_features=enable_research
        )
        stats = controller.generate_suite()
        
        # Summary
        print(f"\n{'='*70}")
        print("SUITE GENERATION SUCCESSFUL")
        print(f"{'='*70}")
        print(f"Total Constraints: {stats['total_constraints']}")
        print(f"Valid: {stats['total_valid']}")
        print(f"SAT: {stats['total_sat']} ({stats['total_sat']/stats['total_constraints']*100:.1f}%)" if stats['total_constraints'] > 0 else "SAT: 0")
        print(f"UNSAT: {stats['total_unsat']} ({stats['total_unsat']/stats['total_constraints']*100:.1f}%)" if stats['total_constraints'] > 0 else "UNSAT: 0")
        print(f"Unknown: {stats['total_unknown']}")
        print(f"\nOutputs saved to: {suite.output_root}")
        
        if enable_research:
            #print(f"\n Advanced Features Applied:")
            print("   Metadata Enrichment")
            print("   UNSAT Generation")
            print("   AST Similarity")
            print("   Semantic Similarity")
            print("   Implication Checking")
            print("   Manifest.jsonl")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
