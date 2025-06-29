#!/usr/bin/env python3
"""
Performance Test for Chess Board Display
Measures the time taken to generate PGNs and display chess positions.
"""

import time
import statistics
from typing import List, Tuple
from pgn_generator import PGNGenerator
from chess_board import ChessBoard

class PerformanceTester:
    """Tests performance of chess board operations."""
    
    def __init__(self):
        self.pgn_generator = PGNGenerator()
        self.chess_board = ChessBoard()
        self.results = []
    
    def test_position_generation_and_display(self, iterations: int = 100, 
                                           show_gui: bool = False) -> List[Tuple[float, str]]:
        """Test the performance of generating PGNs and displaying positions."""
        print(f"Performance Test: {iterations} iterations")
        print("=" * 50)
        
        # Generate positions first
        print("Generating positions...")
        positions = self.pgn_generator.generate_position_set(iterations)
        
        print(f"Generated {len(positions)} positions")
        print("Starting performance test...")
        print()
        
        times = []
        
        for i, position in enumerate(positions, 1):
            start_time = time.perf_counter()
            
            # Set the position
            self.chess_board.set_position(position)
            
            # Display the position (console only for performance)
            self.chess_board.display()
            
            # Optionally show GUI (slower, so only for first few)
            if show_gui and i <= 3:
                self.chess_board.display_gui(f"Position {i}")
                time.sleep(0.5)  # Brief pause to see the GUI
            
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            times.append(elapsed_time)
            self.results.append((elapsed_time, position))
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Completed {i}/{iterations} iterations...")
        
        return self.results
    
    def test_console_only(self, iterations: int = 100) -> List[Tuple[float, str]]:
        """Test performance with console display only (faster)."""
        print(f"Console-Only Performance Test: {iterations} iterations")
        print("=" * 50)
        
        # Generate positions
        positions = self.pgn_generator.generate_position_set(iterations)
        
        times = []
        
        for i, position in enumerate(positions, 1):
            start_time = time.perf_counter()
            
            # Set and display position
            self.chess_board.set_position(position)
            self.chess_board.display()
            
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000
            
            times.append(elapsed_time)
            self.results.append((elapsed_time, position))
            
            if i % 20 == 0:
                print(f"Completed {i}/{iterations} iterations...")
        
        return self.results
    
    def test_gui_only(self, iterations: int = 10) -> List[Tuple[float, str]]:
        """Test performance with GUI display only (slower)."""
        print(f"GUI-Only Performance Test: {iterations} iterations")
        print("=" * 50)
        
        positions = self.pgn_generator.generate_position_set(iterations)
        
        times = []
        
        for i, position in enumerate(positions, 1):
            start_time = time.perf_counter()
            
            # Set position and show GUI
            self.chess_board.set_position(position)
            self.chess_board.display_gui(f"Performance Test - Position {i}")
            
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000
            
            times.append(elapsed_time)
            self.results.append((elapsed_time, position))
            
            print(f"Position {i}: {elapsed_time:.2f} ms")
        
        return self.results
    
    def analyze_results(self) -> None:
        """Analyze and display performance results."""
        if not self.results:
            print("No results to analyze. Run a test first.")
            return
        
        times = [result[0] for result in self.results]
        
        print("\nPerformance Analysis")
        print("=" * 30)
        print(f"Total iterations: {len(times)}")
        print(f"Total time: {sum(times):.2f} ms")
        print(f"Average time: {statistics.mean(times):.2f} ms")
        print(f"Median time: {statistics.median(times):.2f} ms")
        print(f"Min time: {min(times):.2f} ms")
        print(f"Max time: {max(times):.2f} ms")
        print(f"Standard deviation: {statistics.stdev(times):.2f} ms")
        
        # Show fastest and slowest positions
        sorted_results = sorted(self.results, key=lambda x: x[0])
        
        print(f"\nFastest position ({sorted_results[0][0]:.2f} ms):")
        print(f"  {sorted_results[0][1]}")
        
        print(f"\nSlowest position ({sorted_results[-1][0]:.2f} ms):")
        print(f"  {sorted_results[-1][1]}")
        
        # Show some sample positions
        print(f"\nSample positions:")
        for i in range(min(5, len(sorted_results))):
            print(f"  {i+1}. {sorted_results[i][1]} ({sorted_results[i][0]:.2f} ms)")
    
    def run_comprehensive_test(self, iterations: int = 100) -> None:
        """Run a comprehensive performance test."""
        print("Comprehensive Chess Board Performance Test")
        print("=" * 60)
        
        # Test 1: Console only (fastest)
        print("\n1. Testing console display performance...")
        self.test_console_only(iterations)
        self.analyze_results()
        
        # Test 2: With GUI (slower, fewer iterations)
        print(f"\n2. Testing GUI display performance (limited to 5 iterations)...")
        gui_results = self.test_gui_only(5)
        
        gui_times = [result[0] for result in gui_results]
        print(f"\nGUI Performance Summary:")
        print(f"  Average GUI time: {statistics.mean(gui_times):.2f} ms")
        print(f"  GUI is {statistics.mean(gui_times) / statistics.mean([r[0] for r in self.results]):.1f}x slower than console")
        
        # Test 3: Mixed test
        print(f"\n3. Testing mixed console/GUI performance...")
        self.results = []  # Reset results
        self.test_position_generation_and_display(10, show_gui=True)
        
        print(f"\nMixed test completed. GUI windows will remain open.")
        print("Close the GUI windows to continue...")
        
        # Keep GUI open
        if self.chess_board.root:
            self.chess_board.root.mainloop()


def main():
    """Run the performance test."""
    tester = PerformanceTester()
    
    # Run comprehensive test
    tester.run_comprehensive_test(100)


if __name__ == "__main__":
    main() 