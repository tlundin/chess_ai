#!/usr/bin/env python3
"""
Simple Performance Test for Chess Board Display
Measures the time taken to generate PGNs and display chess positions (console only).
"""

import time
import statistics
from typing import List, Tuple
from pgn_generator import PGNGenerator
from chess_board import ChessBoard

def run_performance_test(iterations: int = 100) -> List[Tuple[float, str]]:
    """Run a simple performance test with console display only."""
    print(f"Chess Board Performance Test: {iterations} iterations")
    print("=" * 60)
    
    # Initialize components
    pgn_generator = PGNGenerator()
    chess_board = ChessBoard()
    
    # Generate positions
    print("Generating positions...")
    positions = pgn_generator.generate_position_set(iterations)
    print(f"Generated {len(positions)} positions")
    
    # Run performance test
    print("Starting performance test...")
    results = []
    
    for i, position in enumerate(positions, 1):
        start_time = time.perf_counter()
        
        # Set the position
        chess_board.set_position(position)
        
        # Display the position (console only)
        chess_board.display()
        
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        results.append((elapsed_time, position))
        
        # Progress indicator
        if i % 20 == 0:
            print(f"Completed {i}/{iterations} iterations...")
    
    return results

def analyze_performance(results: List[Tuple[float, str]]) -> None:
    """Analyze and display performance results."""
    if not results:
        print("No results to analyze.")
        return
    
    times = [result[0] for result in results]
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Total iterations: {len(times)}")
    print(f"Total time: {sum(times):.2f} ms")
    print(f"Average time per iteration: {statistics.mean(times):.2f} ms")
    print(f"Median time: {statistics.median(times):.2f} ms")
    print(f"Min time: {min(times):.2f} ms")
    print(f"Max time: {max(times):.2f} ms")
    print(f"Standard deviation: {statistics.stdev(times):.2f} ms")
    
    # Performance categories
    fast_threshold = statistics.mean(times) - statistics.stdev(times)
    slow_threshold = statistics.mean(times) + statistics.stdev(times)
    
    fast_count = sum(1 for t in times if t < fast_threshold)
    slow_count = sum(1 for t in times if t > slow_threshold)
    normal_count = len(times) - fast_count - slow_count
    
    print(f"\nPerformance Distribution:")
    print(f"  Fast (< {fast_threshold:.2f} ms): {fast_count} iterations")
    print(f"  Normal ({fast_threshold:.2f} - {slow_threshold:.2f} ms): {normal_count} iterations")
    print(f"  Slow (> {slow_threshold:.2f} ms): {slow_count} iterations")
    
    # Show fastest and slowest positions
    sorted_results = sorted(results, key=lambda x: x[0])
    
    print(f"\nFastest 3 positions:")
    for i in range(min(3, len(sorted_results))):
        print(f"  {i+1}. {sorted_results[i][1]} ({sorted_results[i][0]:.2f} ms)")
    
    print(f"\nSlowest 3 positions:")
    for i in range(min(3, len(sorted_results))):
        idx = len(sorted_results) - 1 - i
        print(f"  {i+1}. {sorted_results[idx][1]} ({sorted_results[idx][0]:.2f} ms)")

def main():
    """Run the performance test."""
    # Run the test
    results = run_performance_test(100)
    
    # Analyze results
    analyze_performance(results)
    
    print(f"\nTest completed successfully!")
    print(f"Average performance: {statistics.mean([r[0] for r in results]):.2f} ms per position")

if __name__ == "__main__":
    main() 