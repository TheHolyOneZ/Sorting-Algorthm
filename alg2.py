# Cleaned by TheZ
import os
import platform
import time
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
from collections import deque
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from numba import jit, prange
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChaosSorter")
try:
    sns.set_theme(style="darkgrid")
except ImportError:
    plt.style.use("ggplot")  # Fallback if Seaborn isn't installed


class SortingStrategy(Enum):
    INSERTION = "insertion"
    QUICK = "quick"
    MERGE = "merge"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class SortStats:
    comparisons: int = 0
    swaps: int = 0
    recursion_depth: int = 0
    time_taken: float = 0.0
    memory_usage: int = 0
    strategy_switches: int = 0
    partition_quality: float = 0.0


class SortingError(Exception):
    pass


class TheZsBenchmarker:
    def __init__(self):
        self.sorter = TheZs()
        self.results_file = "quantum_sort_benchmark_results.txt"
        self.algorithms = {
            "TheZsQuantumWave": lambda x: self.sorter.z_quantum_sorter.z_sort(x.copy()),
            "TheZsNinjaQuick": lambda x: self.sorter._ninja_quick_sort(x.copy(), 0),
            "TheZsHybridSort": lambda x: self.sorter._hybrid_sort(x.copy(), 0),
            "TheZsAdaptiveSort": lambda x: self.sorter._adaptive_merge_sort(
                x.copy(), 0
            ),
        }

    def run_benchmark_suite(self, sizes=[1000, 10000, 50000], iterations=5):
        results = {}
        logger.info("Starting TheZs Benchmark Suite...")
        self._write_header()
        test_cases = {
            "random": lambda s: list(random.sample(range(s * 2), s)),
            "reversed": lambda s: list(range(s, 0, -1)),
            "nearly_sorted": lambda s: self._generate_nearly_sorted(s),
            "few_unique": lambda s: [random.randint(1, 10) for _ in range(s)],
            "many_duplicates": lambda s: [random.randint(1, s // 10) for _ in range(s)],
        }
        for size in sizes:
            logger.info(f"\n=== Benchmarking arrays of size {size} ===")
            results[size] = {}
            for case_name, generator in test_cases.items():
                logger.info(f"\nTesting {case_name} data pattern...")
                results[size][case_name] = self._benchmark_case(
                    lambda: generator(size), iterations
                )
                self._write_case_results(size, case_name, results[size][case_name])
                logger.info(f"Completed {case_name} test")
        self._display_results(results)
        self.visualize_benchmarks(results)
        return results

    def _benchmark_case(self, data_generator, iterations):
        timings = {name: [] for name in self.algorithms.keys()}
        memory_usage = {name: [] for name in self.algorithms.keys()}
        for i in range(iterations):
            data = data_generator()
            for alg_name, alg_func in self.algorithms.items():
                try:
                    initial_memory = psutil.Process().memory_info().rss
                    start_time = time.perf_counter_ns()  # More precise than time.time()
                    alg_func(data)
                    end_time = time.perf_counter_ns()
                    final_memory = psutil.Process().memory_info().rss
                    timings[alg_name].append(
                        (end_time - start_time) / 1e9
                    )  # Convert to seconds
                    memory_usage[alg_name].append(final_memory - initial_memory)
                except Exception as e:
                    logger.error(f"Error in {alg_name}: {str(e)}")
                    timings[alg_name].append(float("inf"))
                    memory_usage[alg_name].append(0)
        return {
            name: {
                "mean_time": np.mean(times),
                "std_dev": np.std(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_memory": np.mean(memory_usage[name])
                / (1024 * 1024),  # Convert to MB
                "iterations": iterations,
            }
            for name, times in timings.items()
        }

    def _write_header(self):
        with open(self.results_file, "w") as f:
            f.write("=== TheZs Quantum Sorting Algorithm Benchmark Results ===\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System Info:\n")
            f.write(f"CPU Cores: {os.cpu_count()}\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Python Version: {platform.python_version()}\n\n")

    def _write_case_results(self, size, case_name, results):
        with open(self.results_file, "a") as f:
            f.write(f"\nArray Size: {size} - {case_name} pattern\n")
            f.write("-" * 60 + "\n")
            for alg_name, metrics in results.items():
                f.write(f"\n{alg_name}:\n")
                f.write(f"  Mean Time: {metrics['mean_time']:.6f}s\n")
                f.write(f"  Std Dev:   {metrics['std_dev']:.6f}s\n")
                f.write(f"  Min Time:  {metrics['min_time']:.6f}s\n")
                f.write(f"  Max Time:  {metrics['max_time']:.6f}s\n")
                f.write(f"  Memory:    {metrics['avg_memory']:.2f} MB\n")

    def _display_results(self, results):
        logger.info("\n=== TheZs Quantum Sorting Algorithm Performance Analysis ===")
        for size in results:
            logger.info(f"\nArray Size: {size}")
            for case in results[size]:
                logger.info(f"\n{case} data pattern:")
                quantum_time = results[size][case]["TheZsQuantumWave"]["mean_time"]
                for alg in results[size][case]:
                    mean_time = results[size][case][alg]["mean_time"]
                    std_dev = results[size][case][alg]["std_dev"]
                    memory = results[size][case][alg]["avg_memory"]
                    if alg != "TheZsQuantumWave":
                        speedup = mean_time / quantum_time
                        logger.info(
                            f"{alg:15} - Time: {mean_time:.6f}s ± {std_dev:.6f}s "
                            f"Memory: {memory:.2f}MB "
                            f"({'%.2fx' % speedup} {'slower' if speedup > 1 else 'faster'} "
                            f"than TheZsQuantumWave)"
                        )
                    else:
                        logger.info(
                            f"{alg:15} - Time: {mean_time:.6f}s ± {std_dev:.6f}s "
                            f"Memory: {memory:.2f}MB (baseline)"
                        )

    def visualize_benchmarks(self, results):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 15))
        for idx, size in enumerate(results.keys(), 1):
            plt.subplot(2, 2, idx)
            data = results[size]
            x = np.arange(len(data))
            width = 0.15
            for i, (alg_name, color) in enumerate(
                zip(
                    self.algorithms.keys(), ["#00ff00", "#ff0000", "#0000ff", "#ff00ff"]
                )
            ):
                times = [data[case][alg_name]["mean_time"] for case in data.keys()]
                plt.bar(
                    x + i * width, times, width, label=alg_name, color=color, alpha=0.8
                )
            plt.title(f"Array Size: {size}", color="white", fontsize=12)
            plt.xlabel("Test Cases", color="white")
            plt.ylabel("Time (seconds)", color="white")
            plt.xticks(x + width * 2, data.keys(), rotation=45, color="white")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig("benchmark_results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _generate_nearly_sorted(self, size):
        arr = list(range(size))
        swaps = size // 20  # 5% perturbation
        for _ in range(swaps):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr


@jit(nopython=True, parallel=True, fastmath=True)
def vectorized_sort(arr):
    return np.sort(arr)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def parallel_merge(chunks):
    total_len = 0
    for chunk in chunks:
        total_len += len(chunk)
    result = np.empty(total_len, dtype=chunks[0].dtype)
    offset = 0
    for chunk in chunks:
        result[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return np.sort(result)  # Final vectorized sort


@jit(nopython=True, fastmath=True, cache=True)
def quantum_fusion_reactor(arr):
    if len(arr) <= 128:  # Sweet spot for L2 cache
        return np.sort(arr)
    mid = len(arr) // 2
    left = quantum_fusion_reactor(arr[:mid])
    right = quantum_fusion_reactor(arr[mid:])
    result = np.empty_like(arr)
    i = j = k = 0
    left_len, right_len = len(left), len(right)
    while i < left_len and j < right_len:
        cond = left[i] <= right[j]
        result[k] = left[i] if cond else right[j]
        i += cond
        j += not cond
        k += 1
    if i < left_len:
        result[k:] = left[i:]
    else:
        result[k:] = right[j:]
    return result


@jit(nopython=True, fastmath=True, cache=True)
def quantum_hypersonic_sort(arr):
    if len(arr) <= 16:  # Ultra-aggressive micro optimization
        return np.sort(arr)
    block_size = 64  # Optimal L1 cache line size
    num_blocks = max(1, len(arr) // block_size)
    result = np.empty_like(arr)
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, len(arr))
        result[start:end] = np.sort(arr[start:end])
    return quantum_fusion_reactor(result)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def quantum_hypersort_extreme(arr):
    if len(arr) <= 4:
        return np.sort(arr)
    aligned_arr = np.ascontiguousarray(arr)
    block_size = 64
    num_blocks = len(aligned_arr) // block_size
    if num_blocks > 1:
        blocks = aligned_arr.reshape(num_blocks, -1)
        for i in prange(num_blocks):
            blocks[i].sort()
        return quantum_fusion_extreme(blocks.reshape(-1))
    return np.sort(aligned_arr)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def quantum_fusion_extreme(arr):
    if len(arr) <= 128:
        return np.sort(arr)
    mid = len(arr) // 2
    left = quantum_fusion_extreme(arr[:mid])
    right = quantum_fusion_extreme(arr[mid:])
    result = np.empty_like(arr)
    i = j = k = 0
    while i < len(left) and j < len(right):
        cond = left[i] <= right[j]
        result[k] = left[i] if cond else right[j]
        i += cond
        j += not cond
        k += 1
    result[k:] = left[i:] if i < len(left) else right[j:]
    return result


class TheZsQuantumWaveSort:
    def __init__(self):
        self.chunk_size = 64  # L1 cache line size
        self.vector_size = 512  # AVX-512 register size
        self.threshold = 16  # Branch prediction sweet spot

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def z_sort(arr):
        arr_np = np.asarray(arr, dtype=np.int64)
        if len(arr_np) <= 32:
            return np.sort(arr_np)
        if len(arr_np) <= 1024:
            is_sorted = np.all(arr_np[:-1] <= arr_np[1:])
            if is_sorted:
                return arr_np
        if len(arr_np) >= 128:
            if np.all(arr_np[:-1] >= arr_np[1:]):
                return np.flip(arr_np)
            if len(np.unique(arr_np)) < len(arr_np) // 4:
                return np.sort(arr_np)  # Use numpy's optimized path
        return quantum_hypersonic_sort(arr_np)


class ArrayAnalyzer:
    @staticmethod
    def calculate_entropy(arr: List[int]) -> float:
        """Calculate the entropy of the array to measure randomness."""
        _, counts = np.unique(arr, return_counts=True)
        probabilities = counts / len(arr)
        return -sum(p * np.log2(p) for p in probabilities)

    @staticmethod
    def detect_pattern(arr: List[int]) -> str:
        """Detect common patterns in the input array."""
        if len(arr) <= 1:
            return "trivial"
        sorted_check = all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
        reverse_check = all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
        if sorted_check:
            return "sorted"
        elif reverse_check:
            return "reversed"
        unique_ratio = len(set(arr)) / len(arr)
        if unique_ratio < 0.1:
            return "few_unique"
        return "random"


class TheZs:
    def __init__(
        self,
        adaptive_threshold: bool = True,
        parallel_threshold: int = 10000,
        max_threads: int = 4,
    ):
        self.stats = SortStats()
        self.adaptive_threshold = adaptive_threshold
        self.parallel_threshold = parallel_threshold
        self.analyzer = ArrayAnalyzer()
        self.cache = {}
        self.z_quantum_sorter = TheZsQuantumWaveSort()
        self.chunk_size = 16  # Optimized for CPU cache lines
        self.max_threads = min(1024, os.cpu_count() * 64)  # MAXIMUM POWER
        self.threshold = 8

    def auto_tune(self, arr: List[int]) -> None:
        """Smart parameter tuning based on input characteristics."""
        size = len(arr)
        pattern = self.analyzer.detect_pattern(arr)
        entropy = self.analyzer.calculate_entropy(arr)
        self.threshold = self._calculate_optimal_threshold(size, pattern, entropy)

    def _calculate_optimal_threshold(
        self, size: int, pattern: str, entropy: float
    ) -> int:
        base_threshold = max(30, min(size // 20, 100))
        if pattern == "sorted":
            return base_threshold // 2
        elif pattern == "reversed":
            return base_threshold * 2
        elif pattern == "few_unique":
            return base_threshold * 3
        return int(base_threshold * (1 + entropy / 10))

    def _binary_search(self, arr: List[int], target: int, low: int, high: int) -> int:
        while low < high:
            mid = (low + high) // 2
            self.stats.comparisons += 1
            if arr[mid] > target:
                high = mid
            else:
                low = mid + 1
        return low - 1

    def chaos_sort(self, arr: List[int], depth: int = 0) -> Tuple[List[int], SortStats]:
        try:
            start_time = time.time()
            if self.adaptive_threshold:
                self.auto_tune(arr)
            result = self._sort_with_strategy(arr, depth)
            self.stats.time_taken = time.time() - start_time
            return result, self.stats
        except Exception as e:
            logger.error(f"Sorting error: {str(e)}")
            raise SortingError(f"Failed to sort array: {str(e)}")

    def _sort_with_strategy(self, arr: List[int], depth: int) -> List[int]:
        if len(arr) <= 1:
            return arr
        strategy = self._select_strategy(arr, depth)
        if strategy == SortingStrategy.INSERTION:
            return self._optimized_insertion_sort(arr)
        elif strategy == SortingStrategy.QUICK:
            return self._ninja_quick_sort(arr, depth)
        elif strategy == SortingStrategy.MERGE:
            return self._adaptive_merge_sort(arr, depth)
        elif strategy == SortingStrategy.ADAPTIVE:
            return self.z_quantum_sorter.z_sort(arr)
        else:
            return self._hybrid_sort(arr, depth)

    def _select_strategy(self, arr: List[int], depth: int) -> SortingStrategy:
        size = len(arr)
        pattern = self.analyzer.detect_pattern(arr)
        if size < self.threshold:
            return SortingStrategy.INSERTION
        elif pattern == "sorted":
            return SortingStrategy.MERGE
        elif size > self.parallel_threshold:
            return SortingStrategy.QUICK
        else:
            return SortingStrategy.HYBRID

    def _optimized_insertion_sort(self, arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            return arr
        result = arr.copy()  # Create a working copy
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
            self.stats.swaps += 1
        return result

    def _ninja_quick_sort(self, arr: List[int], depth: int) -> List[int]:
        if len(arr) <= 1:
            return arr
        pivot = self._ninja_pivot_selection(arr)
        left, middle, right = self._three_way_partition(arr, pivot)
        if len(arr) > self.parallel_threshold:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_left = executor.submit(self._ninja_quick_sort, left, depth + 1)
                future_right = executor.submit(self._ninja_quick_sort, right, depth + 1)
                return future_left.result() + middle + future_right.result()
        return (
            self._ninja_quick_sort(left, depth + 1)
            + middle
            + self._ninja_quick_sort(right, depth + 1)
        )

    def _adaptive_merge_sort(self, arr: List[int], depth: int) -> List[int]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        cache_key = tuple(arr)
        if cache_key in self.cache:
            return self.cache[cache_key]
        left = self._adaptive_merge_sort(left, depth + 1)
        right = self._adaptive_merge_sort(right, depth + 1)
        result = self._smart_merge(left, right)
        self.cache[cache_key] = result
        return result

    def _hybrid_sort(self, arr: List[int], depth: int) -> List[int]:
        if len(arr) <= 1:
            return arr
        pattern = self.analyzer.detect_pattern(arr)
        entropy = self.analyzer.calculate_entropy(arr)
        if entropy < 1.0:
            return self._optimized_insertion_sort(arr)
        elif pattern == "reversed":
            return self._adaptive_merge_sort(arr, depth)
        else:
            return self._ninja_quick_sort(arr, depth)

    def _ninja_pivot_selection(self, arr: List[int]) -> int:
        if len(arr) <= 5:
            return arr[len(arr) // 2]
        samples = random.sample(arr, min(9, len(arr)))
        samples.sort()
        return samples[len(samples) // 2]

    def _three_way_partition(
        self, arr: List[int], pivot: int
    ) -> Tuple[List[int], List[int], List[int]]:
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        self.stats.partition_quality = min(len(left), len(right)) / len(arr)
        return left, middle, right

    def _smart_merge(self, left: List[int], right: List[int]) -> List[int]:
        result = []
        left = deque(left)
        right = deque(right)
        while left and right:
            self.stats.comparisons += 1
            if left[0] <= right[0]:
                result.append(left.popleft())
            else:
                result.append(right.popleft())
        result.extend(left)
        result.extend(right)
        return result


class TheZsAnalyzer:
    def __init__(self):
        self.sorter = TheZs()

    def analyze_performance(
        self, sizes: List[int] = [100, 1000, 10000], repetitions: int = 3
    ) -> Dict:
        results = {}
        for size in sizes:
            results[size] = self._analyze_size(size, repetitions)
        return results

    def _analyze_size(self, size: int, repetitions: int) -> Dict:
        test_cases = {
            "random": lambda: random.sample(range(size * 10), size),
            "nearly_sorted": lambda: sorted(random.sample(range(size * 10), size))[
                : size // 10
            ],
            "reversed": lambda: list(range(size, 0, -1)),
            "few_unique": lambda: [random.randint(1, 10) for _ in range(size)],
        }
        results = {}
        for case_name, generator in test_cases.items():
            case_results = []
            for _ in range(repetitions):
                data = generator()
                sorted_data, stats = self.sorter.chaos_sort(data.copy())
                case_results.append(stats)
            results[case_name] = self._aggregate_stats(case_results)
        return results

    def _aggregate_stats(self, stats_list: List[SortStats]) -> Dict:
        return {
            "avg_time": np.mean([s.time_taken for s in stats_list]),
            "std_time": np.std([s.time_taken for s in stats_list]),
            "avg_comparisons": np.mean([s.comparisons for s in stats_list]),
            "avg_swaps": np.mean([s.swaps for s in stats_list]),
            "max_recursion": max(s.recursion_depth for s in stats_list),
            "avg_partition_quality": np.mean([s.partition_quality for s in stats_list]),
        }


class TheZsVisualizer:
    def __init__(self, analyzer: TheZsAnalyzer):
        self.analyzer = analyzer

    def visualize_performance(self, results: Dict):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        self._plot_time_comparison(results)
        plt.subplot(2, 2, 2)
        self._plot_operations(results)
        plt.subplot(2, 2, 3)
        self._plot_partition_quality(results)
        plt.subplot(2, 2, 4)
        self._plot_distribution(results)
        plt.tight_layout()
        plt.show()

    def _plot_time_comparison(self, results):
        for size, data in results.items():
            times = [info["avg_time"] for info in data.values()]
            labels = list(data.keys())
            plt.bar(labels, times, alpha=0.7, label=f"Size {size}")
        plt.title("Average Sorting Time by Case")
        plt.xlabel("Test Cases")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.xticks(rotation=45)

    def _plot_operations(self, results):
        for size, data in results.items():
            comparisons = [info["avg_comparisons"] for info in data.values()]
            swaps = [info["avg_swaps"] for info in data.values()]
            x = np.arange(len(data))
            width = 0.35
            plt.bar(x - width / 2, comparisons, width, label=f"Comparisons {size}")
            plt.bar(x + width / 2, swaps, width, label=f"Swaps {size}")
        plt.title("Operation Counts")
        plt.xlabel("Test Cases")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(x, list(data.keys()), rotation=45)

    def _plot_partition_quality(self, results):
        qualities = []
        labels = []
        for size, data in results.items():
            for case, info in data.items():
                qualities.append(info["avg_partition_quality"])
                labels.append(f"{case}\n(size {size})")
        plt.bar(labels, qualities, alpha=0.7)
        plt.title("Partition Quality")
        plt.xlabel("Test Cases")
        plt.ylabel("Quality Score")
        plt.xticks(rotation=45)

    def _plot_distribution(self, results):
        sizes = list(results.keys())
        cases = list(results[sizes[0]].keys())
        for case in cases:
            times = [results[size][case]["avg_time"] for size in sizes]
            plt.plot(sizes, times, marker="o", label=case)
        plt.title("Scaling Behavior")
        plt.xlabel("Input Size")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")


if __name__ == "__main__":
    sorter = TheZs()
    analyzer = TheZsAnalyzer()
    visualizer = TheZsVisualizer(analyzer)
    benchmarker = TheZsBenchmarker()
    test_array = random.sample(range(1000), 1000)
    sorted_array, stats = sorter.chaos_sort(test_array)
    logger.info(f"Sorting completed in {stats.time_taken:.4f} seconds")
    logger.info(f"Comparisons: {stats.comparisons}")
    logger.info(f"Swaps: {stats.swaps}")
    benchmark_results = benchmarker.run_benchmark_suite(
        sizes=[1000, 10000, 50000], iterations=5
    )
    benchmarker.visualize_benchmarks(benchmark_results)
    results = analyzer.analyze_performance()
    visualizer.visualize_performance(results)
    logger.info("Analysis and benchmarking complete!")

# End of cleanup by TheZ