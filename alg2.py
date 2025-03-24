
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
from numba import jit, prange, cuda


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChaosSorter")


try:
    sns.set_theme(style="darkgrid")
except ImportError:
    plt.style.use("ggplot")  


class SortingStrategy(Enum):
    INSERTION = "insertion"
    QUICK = "quick"
    MERGE = "merge"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    QUANTUM = "quantum"
    HYPERSONIC = "hypersonic"


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

@jit(nopython=True, fastmath=True, cache=True)
def insertion_sort_jit(arr):
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


@jit(nopython=True, fastmath=True, cache=True)
def quick_sort_partition(arr, low, high):
    
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


@jit(nopython=True, fastmath=True, cache=True)
def quick_sort_jit(arr, low, high):
    
    if len(arr) <= 1:
        return arr
    if low < high:
        pi = quick_sort_partition(arr, low, high)
        quick_sort_jit(arr, low, pi - 1)
        quick_sort_jit(arr, pi + 1, high)
    return arr


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def merge_jit(left, right):
    
    result = np.empty(len(left) + len(right), dtype=left.dtype)
    i = j = k = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1
        
    while i < len(left):
        result[k] = left[i]
        i += 1
        k += 1
        
    while j < len(right):
        result[k] = right[j]
        j += 1
        k += 1
        
    return result


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def merge_sort_jit(arr):
    
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort_jit(arr[:mid])
    right = merge_sort_jit(arr[mid:])
    
    return merge_jit(left, right)



@jit(nopython=True, parallel=True, fastmath=True)
def vectorized_sort(arr):
    
    return np.sort(arr)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def parallel_merge(chunks):
    
    total_len = sum(len(chunk) for chunk in chunks)
    result = np.empty(total_len, dtype=chunks[0].dtype)
    
    offset = 0
    for chunk in chunks:
        result[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
        

    return quick_sort_jit(result, 0, len(result) - 1)



@jit(nopython=True, fastmath=True, cache=True)
def quantum_fusion_reactor(arr):
    
    if len(arr) <= 128:
        return insertion_sort_jit(arr) if len(arr) <= 16 else quick_sort_jit(arr, 0, len(arr) - 1)
    
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
    
    if len(arr) <= 16:
        return insertion_sort_jit(arr)
    

    block_size = min(64, max(16, len(arr) // 32))
    num_blocks = max(1, len(arr) // block_size)
    result = np.empty_like(arr)
    

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, len(arr))
        result[start:end] = quick_sort_jit(arr[start:end].copy(), 0, end - start - 1)
    

    if len(arr) % block_size != 0:
        start = num_blocks * block_size
        result[start:] = quick_sort_jit(arr[start:].copy(), 0, len(arr) - start - 1)
    
    return quantum_fusion_reactor(result)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def quantum_hypersort_extreme(arr):
    
    if len(arr) <= 4:
        return insertion_sort_jit(arr)
    
    aligned_arr = np.ascontiguousarray(arr)
    block_size = min(128, max(32, len(aligned_arr) // 16))
    num_blocks = max(1, len(aligned_arr) // block_size)
    
    if num_blocks > 1:

        for i in prange(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(aligned_arr))
            aligned_arr[start:end] = quick_sort_jit(aligned_arr[start:end].copy(), 0, end - start - 1)
        
        return quantum_fusion_extreme(aligned_arr)
    
    return quick_sort_jit(aligned_arr, 0, len(aligned_arr) - 1)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def quantum_fusion_extreme(arr):
    
    if len(arr) <= 128:
        return quick_sort_jit(arr, 0, len(arr) - 1)
    
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
    

    if i < len(left):
        result[k:] = left[i:]
    else:
        result[k:] = right[j:]
    
    return result


@jit(nopython=True, fastmath=True, cache=True)
def quantum_wave_ultra(arr):
    

    if len(arr) <= 16:
        return insertion_sort_jit(arr)
        

    check_size = min(100, len(arr))
    is_sorted = True
    is_reverse_sorted = True
    

    for i in range(check_size - 1):
        if arr[i] > arr[i + 1]:
            is_sorted = False
            if not is_reverse_sorted:  
                break
        if arr[i] < arr[i + 1]:
            is_reverse_sorted = False
            if not is_sorted:  
                break
    

    if is_sorted and check_size == len(arr):
        return arr
    if is_reverse_sorted and check_size == len(arr):
        return arr[::-1]
    

    if (is_sorted or is_reverse_sorted) and len(arr) > check_size:

        step = max(1, (len(arr) - check_size) // 10)
        for i in range(check_size, len(arr), step):
            if i > 0:
                if is_sorted and arr[i-1] > arr[i]:
                    is_sorted = False
                if is_reverse_sorted and arr[i-1] < arr[i]:
                    is_reverse_sorted = False
                if not is_sorted and not is_reverse_sorted:
                    break
        
        if is_sorted:
            return arr
        if is_reverse_sorted:
            return arr[::-1]
    
    if len(arr) > 100:
      
        samples = np.empty(100, dtype=arr.dtype)
        step = max(1, len(arr) // 100)
        for i in range(100):
            samples[i] = arr[min(i * step, len(arr) - 1)]
        
        unique_ratio = len(np.unique(samples)) / len(samples)
        if unique_ratio < 0.1: 
            return counting_sort_jit(arr)
    
    if len(arr) <= 64:
        return insertion_sort_jit(arr)
    elif len(arr) <= 512:
      
        result = arr.copy()
        return quick_sort_jit(result, 0, len(result) - 1)
    elif len(arr) <= 8192:
        
        return quantum_hypersonic_sort(arr)
    else:
       
        return quantum_hypersort_extreme(arr)

@jit(nopython=True, fastmath=True, cache=True)
def counting_sort_jit(arr):
   
    if len(arr) == 0:
        return arr
        
    min_val = arr[0]
    max_val = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] < min_val:
            min_val = arr[i]
        elif arr[i] > max_val:
            max_val = arr[i]
    
    range_size = max_val - min_val + 1
    counts = np.zeros(range_size, dtype=np.int64)
    
    for i in range(len(arr)):
        counts[arr[i] - min_val] += 1
    
    result = np.empty_like(arr)
    pos = 0
    
    for i in range(range_size):
        for j in range(counts[i]):
            result[pos] = i + min_val
            pos += 1
    
    return result



try:
    @cuda.jit
    def cuda_sort_kernel(arr, result):
        
        i = cuda.grid(1)
        if i < len(arr):

            for j in range(1, len(arr)):
                key = arr[j]
                k = j - 1
                while k >= 0 and arr[k] > key:
                    arr[k + 1] = arr[k]
                    k -= 1
                arr[k + 1] = key
            result[i] = arr[i]
    
    def cuda_sort(arr):
        
        d_arr = cuda.to_device(arr)
        d_result = cuda.device_array_like(d_arr)
        

        threads_per_block = 256
        blocks_per_grid = (len(arr) + threads_per_block - 1) // threads_per_block
        

        cuda_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_result)
        

        return d_result.copy_to_host()
except:

    def cuda_sort(arr):
        return quantum_wave_ultra(arr)


class TheZsQuantumWaveSort:
    
    def __init__(self):
        self.chunk_size = 128  
        self.vector_size = 1024  
        self.threshold = 32  
        self.use_cuda = False
        

        try:
            cuda.detect()
            self.use_cuda = True
            logger.info("CUDA detected and enabled for sorting")
        except:
            logger.info("CUDA not available, using CPU optimized algorithms")

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def z_sort(arr):
        
        arr_np = np.asarray(arr, dtype=np.int64)
        

        if len(arr_np) <= 32:
            return insertion_sort_jit(arr_np)
            
        if len(arr_np) <= 1024:
            is_sorted = True
            for i in range(len(arr_np) - 1):
                if arr_np[i] > arr_np[i + 1]:
                    is_sorted = False
                    break
            if is_sorted:
                return arr_np
                

        if len(arr_np) >= 128:
            is_reverse = True
            for i in range(len(arr_np) - 1):
                if arr_np[i] < arr_np[i + 1]:
                    is_reverse = False
                    break
            if is_reverse:
                return arr_np[::-1]
                

            if len(arr_np) > 100:
                sample = arr_np[:100]
                unique_count = len(np.unique(sample))
                if unique_count < len(sample) // 4:
                    return counting_sort_jit(arr_np)
        
        return quantum_wave_ultra(arr_np)
    
    def sort(self, arr):
        
        if self.use_cuda and len(arr) > self.vector_size:
            try:
                return cuda_sort(np.array(arr, dtype=np.int64))
            except:

                return self.z_sort(arr)
        else:
            return self.z_sort(arr)


class ArrayAnalyzer:
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def calculate_entropy(arr):
        
        if len(arr) <= 1:
            return 0.0
            

        if len(arr) > 1000:
            indices = np.linspace(0, len(arr)-1, 1000, dtype=np.int64)
            sample = arr[indices]
        else:
            sample = arr
            

        unique_vals = {}
        for val in sample:
            if val in unique_vals:
                unique_vals[val] += 1
            else:
                unique_vals[val] = 1
                

        entropy = 0.0
        n = len(sample)
        for count in unique_vals.values():
            p = count / n
            entropy -= p * np.log2(p)
            
        return entropy

    @staticmethod
    def detect_pattern(arr):
        
        if len(arr) <= 1:
            return "trivial"
            

        if len(arr) > 1000:
            check_size = 1000
            step = len(arr) // check_size
            sample_indices = [i * step for i in range(check_size)]
            sample = [arr[i] for i in sample_indices]
        else:
            sample = arr
            

        sorted_check = True
        for i in range(len(sample) - 1):
            if sample[i] > sample[i + 1]:
                sorted_check = False
                break
                
        if sorted_check:
            return "sorted"
            

        reverse_check = True
        for i in range(len(sample) - 1):
            if sample[i] < sample[i + 1]:
                reverse_check = False
                break
                
        if reverse_check:
            return "reversed"
            

        unique_count = len(set(sample))
        unique_ratio = unique_count / len(sample)
        
        if unique_ratio < 0.1:
            return "few_unique"
        elif unique_ratio < 0.3:
            return "some_duplicates"
            

        inversions = 0
        max_check = min(len(sample), 100)
        for i in range(max_check - 1):
            if sample[i] > sample[i + 1]:
                inversions += 1
                
        if inversions < max_check * 0.1:
            return "nearly_sorted"
            
        return "random"


class TheZs:
    
    def __init__(
        self,
        adaptive_threshold: bool = True,
        parallel_threshold: int = 5000,  
        max_threads: int = min(16, os.cpu_count() or 4),  
    ):
        self.stats = SortStats()
        self.adaptive_threshold = adaptive_threshold
        self.parallel_threshold = parallel_threshold
        self.analyzer = ArrayAnalyzer()
        self.cache = {}
        self.z_quantum_sorter = TheZsQuantumWaveSort()
        self.chunk_size = 32  
        self.max_threads = max_threads
        self.threshold = 16  
        

        self.cpu_count = os.cpu_count() or 4
        self.memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  
        

        self._auto_tune_for_hardware()

    def _auto_tune_for_hardware(self):
        

        if self.cpu_count > 8:
            self.max_threads = min(self.cpu_count - 2, 32)  
        else:
            self.max_threads = max(1, self.cpu_count - 1)
            

        if self.memory_available > 8:  
            self.parallel_threshold = 2000
            self.chunk_size = 64
        elif self.memory_available < 2:  
            self.parallel_threshold = 10000
            self.chunk_size = 16
            
        logger.info(f"Auto-tuned for {self.cpu_count} cores and {self.memory_available:.1f}GB RAM")
        logger.info(f"Using {self.max_threads} threads, parallel threshold: {self.parallel_threshold}")

    def auto_tune(self, arr: List[int]) -> None:
        
        size = len(arr)
        pattern = self.analyzer.detect_pattern(arr)
        

        if pattern not in ["sorted", "reversed", "trivial"]:
            entropy = self.analyzer.calculate_entropy(arr)
            self.threshold = self._calculate_optimal_threshold(size, pattern, entropy)
        else:

            if pattern == "sorted":
                self.threshold = 8
            elif pattern == "reversed":
                self.threshold = 8
            else:
                self.threshold = 16
                

        if size > 100000:
            self.parallel_threshold = max(1000, size // 100)
        elif size < 1000:
            self.parallel_threshold = float('inf')  

    def _calculate_optimal_threshold(self, size: int, pattern: str, entropy: float) -> int:
        

        base_threshold = max(16, min(size // 32, 128))
        

        if pattern == "sorted":
            return base_threshold // 4
        elif pattern == "reversed":
            return base_threshold // 4
        elif pattern == "nearly_sorted":
            return base_threshold // 2
        elif pattern == "few_unique":
            return base_threshold * 2
        elif pattern == "some_duplicates":
            return base_threshold * 1.5
            

        entropy_factor = max(0.5, min(2.0, entropy / 4))
        return int(base_threshold * entropy_factor)

    def chaos_sort(self, arr: List[int], depth: int = 0) -> Tuple[List[int], SortStats]:
        
        try:

            self.stats = SortStats()
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            

            if self.adaptive_threshold:
                self.auto_tune(arr)
                

            result = self._sort_with_strategy(arr, depth)
            

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            self.stats.time_taken = end_time - start_time
            self.stats.memory_usage = end_memory - start_memory
            
            return result, self.stats
        except Exception as e:
            logger.error(f"Sorting error: {str(e)}")
            raise SortingError(f"Failed to sort array: {str(e)}")

    def _sort_with_strategy(self, arr: List[int], depth: int) -> List[int]:
        
        if len(arr) <= 1:
            return arr
            

        if len(arr) > 50000:
            return self.z_quantum_sorter.sort(arr)
            
        strategy = self._select_strategy(arr, depth)
        self.stats.recursion_depth = max(self.stats.recursion_depth, depth)
        
        if strategy == SortingStrategy.INSERTION:
            return self._optimized_insertion_sort(arr)
        elif strategy == SortingStrategy.QUICK:
            return self._ninja_quick_sort(arr, depth)
        elif strategy == SortingStrategy.MERGE:
            return self._adaptive_merge_sort(arr, depth)
        elif strategy == SortingStrategy.QUANTUM:
            return self.z_quantum_sorter.sort(arr)
        elif strategy == SortingStrategy.HYPERSONIC:
            return quantum_hypersonic_sort(np.array(arr, dtype=np.int64))
        else:
            return self._hybrid_sort(arr, depth)

    def _select_strategy(self, arr: List[int], depth: int) -> SortingStrategy:
        
        size = len(arr)
        pattern = self.analyzer.detect_pattern(arr)
        

        if size < self.threshold:
            return SortingStrategy.INSERTION
            

        if pattern == "sorted" or pattern == "nearly_sorted":
            return SortingStrategy.MERGE
        elif pattern == "reversed":

            arr.reverse()
            self.stats.swaps += size // 2
            return SortingStrategy.INSERTION
        elif pattern == "few_unique":

            return SortingStrategy.QUANTUM
            

        if size > self.parallel_threshold:
            if size > 100000:
                return SortingStrategy.QUANTUM
            else:
                return SortingStrategy.QUICK
                

        if size > 1000:
            return SortingStrategy.HYPERSONIC
            

        return SortingStrategy.HYBRID

    def _optimized_insertion_sort(self, arr: List[int]) -> List[int]:
        
        if len(arr) <= 1:
            return arr
            
        result = arr.copy()
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            

            if j > 10:  
                insertion_point = self._binary_search(result, key, 0, j)

                if insertion_point < j:
                    temp = result[i]
                    for k in range(i, insertion_point, -1):
                        result[k] = result[k-1]
                    result[insertion_point] = temp
                    self.stats.swaps += 1
            else:

                while j >= 0 and result[j] > key:
                    result[j + 1] = result[j]
                    j -= 1
                    self.stats.comparisons += 1
                result[j + 1] = key
                self.stats.swaps += 1
                
        return result

    def _binary_search(self, arr: List[int], target: int, low: int, high: int) -> int:
        
        while low < high:
            mid = (low + high) // 2
            self.stats.comparisons += 1
            if arr[mid] > target:
                high = mid
            else:
                low = mid + 1
        return low

    def _ninja_quick_sort(self, arr: List[int], depth: int) -> List[int]:
        
        if len(arr) <= 1:
            return arr
            

        if len(arr) <= self.threshold:
            return self._optimized_insertion_sort(arr)
            

        pivot = self._ninja_pivot_selection(arr)
        

        left, middle, right = self._three_way_partition(arr, pivot)
        

        if len(arr) > self.parallel_threshold and depth < 3:  
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_left = executor.submit(self._ninja_quick_sort, left, depth + 1)
                future_right = executor.submit(self._ninja_quick_sort, right, depth + 1)
                sorted_left = future_left.result()
                sorted_right = future_right.result()
                self.stats.strategy_switches += 1
                return sorted_left + middle + sorted_right
        else:

            return (
                self._ninja_quick_sort(left, depth + 1)
                + middle
                + self._ninja_quick_sort(right, depth + 1)
            )

    def _adaptive_merge_sort(self, arr: List[int], depth: int) -> List[int]:
        
        if len(arr) <= 1:
            return arr
            

        if len(arr) <= self.threshold:
            return self._optimized_insertion_sort(arr)
            

        if len(arr) < 1000:
            cache_key = tuple(arr)
            if cache_key in self.cache:
                return self.cache[cache_key]
                

        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        

        if len(arr) > self.parallel_threshold and depth < 3:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_left = executor.submit(self._adaptive_merge_sort, left, depth + 1)
                future_right = executor.submit(self._adaptive_merge_sort, right, depth + 1)
                sorted_left = future_left.result()
                sorted_right = future_right.result()
                result = self._smart_merge(sorted_left, sorted_right)
                self.stats.strategy_switches += 1
        else:

            sorted_left = self._adaptive_merge_sort(left, depth + 1)
            sorted_right = self._adaptive_merge_sort(right, depth + 1)
            result = self._smart_merge(sorted_left, sorted_right)
            

        if len(arr) < 1000:
            cache_key = tuple(arr)
            self.cache[cache_key] = result
            
        return result

    def _hybrid_sort(self, arr: List[int], depth: int) -> List[int]:
        
        if len(arr) <= 1:
            return arr
            

        if len(arr) <= self.threshold:
            return self._optimized_insertion_sort(arr)
            

        pattern = self.analyzer.detect_pattern(arr)
        

        if len(arr) > 1000:
            sample_size = min(1000, len(arr) // 10)
            sample_indices = [random.randint(0, len(arr) - 1) for _ in range(sample_size)]
            sample = [arr[i] for i in sample_indices]
            entropy = self.analyzer.calculate_entropy(sample)
        else:
            entropy = self.analyzer.calculate_entropy(arr)
            

        if pattern == "sorted" or pattern == "nearly_sorted":
            return arr if pattern == "sorted" else self._adaptive_merge_sort(arr, depth)
        elif pattern == "reversed":
            return arr[::-1]  
        elif pattern == "few_unique" or entropy < 1.0:
            return self.z_quantum_sorter.sort(arr)  
        elif len(arr) > self.parallel_threshold:
            return self._ninja_quick_sort(arr, depth)  
        else:

            return self._adaptive_merge_sort(arr, depth)

    def _ninja_pivot_selection(self, arr: List[int]) -> int:
        
        if len(arr) <= 5:
            return arr[len(arr) // 2]
            

        if len(arr) <= 100:
            first = arr[0]
            middle = arr[len(arr) // 2]
            last = arr[-1]
            return sorted([first, middle, last])[1]
            

        samples_count = min(9, len(arr) // 100 + 1)
        step = len(arr) // samples_count
        samples = [arr[i * step] for i in range(samples_count)]
        samples.sort()
        
        return samples[len(samples) // 2]

    def _three_way_partition(self, arr: List[int], pivot: int) -> Tuple[List[int], List[int], List[int]]:
        
        left = []
        middle = []
        right = []
        

        if len(arr) <= 1000:
            for x in arr:
                self.stats.comparisons += 1
                if x < pivot:
                    left.append(x)
                elif x > pivot:
                    right.append(x)
                else:
                    middle.append(x)
        else:

            arr_np = np.array(arr)
            left = arr_np[arr_np < pivot].tolist()
            middle = arr_np[arr_np == pivot].tolist()
            right = arr_np[arr_np > pivot].tolist()
            self.stats.comparisons += len(arr)
            

        if len(arr) > 0:
            self.stats.partition_quality = min(len(left), len(right)) / len(arr)
            
        return left, middle, right

    def _smart_merge(self, left: List[int], right: List[int]) -> List[int]:
        

        if not left:
            return right
        if not right:
            return left
            

        if len(left) + len(right) <= 100:
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                self.stats.comparisons += 1
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
            

        result = []
        left_deque = deque(left)
        right_deque = deque(right)
        
        while left_deque and right_deque:
            self.stats.comparisons += 1
            if left_deque[0] <= right_deque[0]:
                result.append(left_deque.popleft())
            else:
                result.append(right_deque.popleft())
                

        if left_deque:
            result.extend(left_deque)
        if right_deque:
            result.extend(right_deque)
            
        return result


class TheZsBenchmarker:
    
    def __init__(self):
        self.sorter = TheZs()
        self.results_file = "quantum_sort_benchmark_results.txt"
        self.algorithms = {
            "TheZsQuantumWave": lambda x: self.sorter.z_quantum_sorter.sort(x.copy()),
            "TheZsNinjaQuick": lambda x: self.sorter._ninja_quick_sort(x.copy(), 0),
            "TheZsHybridSort": lambda x: self.sorter._hybrid_sort(x.copy(), 0),
            "TheZsAdaptiveSort": lambda x: self.sorter._adaptive_merge_sort(x.copy(), 0),
            "PythonBuiltIn": lambda x: sorted(x.copy()),
            "NumpySort": lambda x: np.sort(np.array(x, dtype=np.int64)).tolist(),
        }

    def run_benchmark_suite(self, sizes=[1000, 10000, 50000, 100000], iterations=5):
        
        results = {}
        logger.info("Starting TheZs Benchmark Suite...")
        self._write_header()
        

        test_cases = {
            "random": lambda s: list(random.sample(range(s * 2), s)),
            "reversed": lambda s: list(range(s, 0, -1)),
            "nearly_sorted": lambda s: self._generate_nearly_sorted(s),
            "few_unique": lambda s: [random.randint(1, 10) for _ in range(s)],
            "many_duplicates": lambda s: [random.randint(1, s // 10) for _ in range(s)],
            "sorted": lambda s: list(range(s)),
            "sawtooth": lambda s: [(i % 10) for i in range(s)],
            "random_with_duplicates": lambda s: [random.randint(1, s // 2) for _ in range(s)],
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
        correctness = {name: True for name in self.algorithms.keys()}
        
        for i in range(iterations):
            data = data_generator()
            correct_result = sorted(data.copy())  
            
            for alg_name, alg_func in self.algorithms.items():
                try:

                    initial_memory = psutil.Process().memory_info().rss
                    start_time = time.perf_counter_ns()
                    result = alg_func(data)
                    end_time = time.perf_counter_ns()
                    final_memory = psutil.Process().memory_info().rss
                    

                    timings[alg_name].append((end_time - start_time) / 1e9)
                    memory_usage[alg_name].append(final_memory - initial_memory)
                    

                    if not self._is_sorted(result) or len(result) != len(data):
                        correctness[alg_name] = False
                        logger.warning(f"{alg_name} failed to sort correctly!")
                except Exception as e:
                    logger.error(f"Error in {alg_name}: {str(e)}")
                    timings[alg_name].append(float("inf"))
                    memory_usage[alg_name].append(0)
                    correctness[alg_name] = False
                    
        return {
            name: {
                "mean_time": np.mean(times),
                "std_dev": np.std(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_memory": np.mean(memory_usage[name]) / (1024 * 1024),  
                "iterations": iterations,
                "correct": correctness[name],
            }
            for name, times in timings.items()
        }

    def _is_sorted(self, arr):
        
        return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

    def _write_header(self):
        
        with open(self.results_file, "w") as f:
            f.write("=== TheZs Quantum Sorting Algorithm Benchmark Results ===\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System Info:\n")
            f.write(f"CPU: {platform.processor()}\n")
            f.write(f"CPU Cores: {os.cpu_count()}\n")
            f.write(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Python Version: {platform.python_version()}\n")
            f.write(f"NumPy Version: {np.__version__}\n\n")

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
                f.write(f"  Correct:   {'Yes' if metrics['correct'] else 'No'}\n")

    def _display_results(self, results):
        
        logger.info("\n=== TheZs Quantum Sorting Algorithm Performance Analysis ===")
        
        for size in results:
            logger.info(f"\nArray Size: {size}")
            
            for case in results[size]:
                logger.info(f"\n{case} data pattern:")
                

                fastest_alg = min(
                    results[size][case].items(),
                    key=lambda x: x[1]["mean_time"] if x[1]["correct"] else float('inf')
                )[0]
                fastest_time = results[size][case][fastest_alg]["mean_time"]
                

                for alg in results[size][case]:
                    mean_time = results[size][case][alg]["mean_time"]
                    std_dev = results[size][case][alg]["std_dev"]
                    memory = results[size][case][alg]["avg_memory"]
                    correct = results[size][case][alg]["correct"]
                    
                    if not correct:
                        status = "FAILED"
                    else:
                        relative_speed = mean_time / fastest_time
                        if alg == fastest_alg:
                            status = "FASTEST"
                        elif relative_speed < 1.1:
                            status = "COMPETITIVE"
                        elif relative_speed < 2:
                            status = f"{relative_speed:.2f}x slower"
                        else:
                            status = f"{relative_speed:.2f}x SLOWER"
                            
                    logger.info(
                        f"{alg:20} - Time: {mean_time:.6f}s ± {std_dev:.6f}s "
                        f"Memory: {memory:.2f}MB "
                        f"Status: {status}"
                    )

    def visualize_benchmarks(self, results):
        
        plt.style.use("dark_background")
        

        fig = plt.figure(figsize=(20, 15))
        

        for idx, size in enumerate(results.keys(), 1):
            plt.subplot(2, 2, idx)
            data = results[size]
            x = np.arange(len(data))
            width = 0.15
            

            colors = {
            "TheZsQuantumWave": "#ff00ff",
            "TheZsNinjaQuick": "#ffff00",
            "TheZsHybridSort": "#00ffff",
            "TheZsAdaptiveSort": "#ff00ff", 
            "PythonBuiltIn": "#ffff00",     
            "NumpySort": "#00ffff"
            }

            
            for i, alg_name in enumerate(self.algorithms.keys()):
                if alg_name in data[list(data.keys())[0]]:  
                    times = [data[case][alg_name]["mean_time"] for case in data.keys()]
                    plt.bar(
                        x + i * width, times, width, 
                        label=alg_name, 
                        color=colors.get(alg_name, f"C{i}"), 
                        alpha=0.8
                    )
            
            plt.title(f"Array Size: {size}", color="white", fontsize=14)
            plt.xlabel("Test Cases", color="white")
            plt.ylabel("Time (seconds)", color="white")
            plt.xticks(x + width * (len(self.algorithms) / 2 - 0.5), data.keys(), rotation=45, color="white")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig("benchmark_results.png", dpi=300, bbox_inches="tight")
        

        plt.figure(figsize=(15, 10))
        

        plt.subplot(2, 1, 1)
        for alg_name in self.algorithms.keys():
            sizes = list(results.keys())

            times = [results[size]["random"][alg_name]["mean_time"] for size in sizes]
            plt.plot(sizes, times, marker='o', linewidth=2, label=alg_name, color=colors.get(alg_name))
            
        plt.title("Algorithm Scaling (Random Data)", color="white", fontsize=14)
        plt.xlabel("Array Size", color="white")
        plt.ylabel("Time (seconds)", color="white")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        

        plt.subplot(2, 1, 2)
        for alg_name in self.algorithms.keys():
            sizes = list(results.keys())
            memory = [results[size]["random"][alg_name]["avg_memory"] for size in sizes]
            plt.plot(sizes, memory, marker='s', linewidth=2, label=alg_name, color=colors.get(alg_name))
            
        plt.title("Memory Usage (Random Data)", color="white", fontsize=14)
        plt.xlabel("Array Size", color="white")
        plt.ylabel("Memory (MB)", color="white")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        
        plt.tight_layout()
        plt.savefig("benchmark_scaling.png", dpi=300, bbox_inches="tight")
        plt.show()
        

        plt.figure(figsize=(15, 10))
        

        largest_size = max(results.keys())
        data = results[largest_size]
        

        algorithms = list(self.algorithms.keys())
        test_cases = list(data.keys())
        
        heatmap_data = np.zeros((len(algorithms), len(test_cases)))
        for i, alg in enumerate(algorithms):
            for j, case in enumerate(test_cases):
                if data[case][alg]["correct"]:
                    heatmap_data[i, j] = data[case][alg]["mean_time"]
                else:
                    heatmap_data[i, j] = np.nan  
        

        plt.title(f"Algorithm Performance Heatmap (Size {largest_size})", color="white", fontsize=16)
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".4f", 
            xticklabels=test_cases, 
            yticklabels=algorithms,
            cmap="viridis_r"  
        )
        plt.tight_layout()
        plt.savefig("benchmark_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _generate_nearly_sorted(self, size):
        
        arr = list(range(size))
        swaps = size // 20  
        

        for _ in range(swaps):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
            
        return arr


class TheZsAnalyzer:
    
    def __init__(self):
        self.sorter = TheZs()

    def analyze_performance(self, sizes: List[int] = [100, 1000, 10000, 100000], repetitions: int = 3) -> Dict:
        
        results = {}
        logger.info("Starting performance analysis...")
        
        for size in sizes:
            logger.info(f"Analyzing arrays of size {size}...")
            results[size] = self._analyze_size(size, repetitions)
            
        return results

    def _analyze_size(self, size: int, repetitions: int) -> Dict:
        

        test_cases = {
            "random": lambda: random.sample(range(size * 10), size),
            "nearly_sorted": lambda: self._generate_nearly_sorted(size, 0.05),  
            "reversed": lambda: list(range(size, 0, -1)),
            "few_unique": lambda: [random.randint(1, 10) for _ in range(size)],
            "many_duplicates": lambda: [random.randint(1, size // 10) for _ in range(size)],
            "sorted": lambda: list(range(size)),
            "sawtooth": lambda: [(i % 10) for i in range(size)],
            "plateau": lambda: [min(i, 10) for i in range(size)],
        }
        
        results = {}
        for case_name, generator in test_cases.items():
            logger.info(f"  Testing {case_name} pattern...")
            case_results = []
            
            for i in range(repetitions):
                data = generator()
                try:
                    sorted_data, stats = self.sorter.chaos_sort(data.copy())

                    is_correct = all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))
                    if not is_correct:
                        logger.warning(f"Incorrect sort detected for {case_name}, iteration {i+1}")
                        stats.time_taken = float('inf')  
                    case_results.append(stats)
                except Exception as e:
                    logger.error(f"Error sorting {case_name}: {str(e)}")

                    dummy_stats = SortStats()
                    dummy_stats.time_taken = float('inf')
                    case_results.append(dummy_stats)
            
            results[case_name] = self._aggregate_stats(case_results)
            
        return results

    def _generate_nearly_sorted(self, size, disorder_ratio):
        
        arr = list(range(size))
        swaps = int(size * disorder_ratio)
        
        for _ in range(swaps):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
            
        return arr

    def _aggregate_stats(self, stats_list: List[SortStats]) -> Dict:
        

        valid_stats = [s for s in stats_list if s.time_taken != float('inf')]
        
        if not valid_stats:
            return {
                "avg_time": float('inf'),
                "std_time": 0,
                "avg_comparisons": 0,
                "avg_swaps": 0,
                "max_recursion": 0,
                "avg_partition_quality": 0,
                "success_rate": 0,
            }
            
        return {
            "avg_time": np.mean([s.time_taken for s in valid_stats]),
            "std_time": np.std([s.time_taken for s in valid_stats]),
            "avg_comparisons": np.mean([s.comparisons for s in valid_stats]),
            "avg_swaps": np.mean([s.swaps for s in valid_stats]),
            "max_recursion": max(s.recursion_depth for s in valid_stats),
            "avg_partition_quality": np.mean([s.partition_quality for s in valid_stats]),
            "success_rate": len(valid_stats) / len(stats_list),
        }


class TheZsVisualizer:
    
    def __init__(self, analyzer: TheZsAnalyzer):
        self.analyzer = analyzer

    def visualize_performance(self, results: Dict):
        
        plt.style.use("dark_background")
        

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        self._plot_time_comparison(results)
        
        plt.subplot(2, 2, 2)
        self._plot_operations(results)
        
        plt.subplot(2, 2, 3)
        self._plot_partition_quality(results)
        
        plt.subplot(2, 2, 4)
        self._plot_scaling(results)
        
        plt.tight_layout()
        plt.savefig("sorting_performance_analysis.png", dpi=300, bbox_inches="tight")
        

        plt.figure(figsize=(20, 15))
        

        plt.subplot(2, 2, 1)
        self._plot_success_rates(results)
        

        plt.subplot(2, 2, 2)
        self._plot_time_distribution(results)
        

        plt.subplot(2, 2, 3)
        self._plot_efficiency(results)
        

        plt.subplot(2, 2, 4)
        self._plot_recursion_depth(results)
        
        plt.tight_layout()
        plt.savefig("sorting_detailed_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_time_comparison(self, results):
        
        sizes = list(results.keys())
        mid_size = sizes[len(sizes)//2]  
        
        data = results[mid_size]
        cases = list(data.keys())
        times = [data[case]["avg_time"] for case in cases]
        
        bars = plt.bar(cases, times, alpha=0.7, color='cyan')
        

        fastest_idx = times.index(min(times))
        slowest_idx = times.index(max(times))
        bars[fastest_idx].set_color('green')
        bars[slowest_idx].set_color('red')
        
        plt.title(f"Average Sorting Time by Case (Size {mid_size})")
        plt.xlabel("Test Cases")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}',
                    ha='center', va='bottom', rotation=0, color='white')

    def _plot_operations(self, results):
        
        sizes = list(results.keys())
        mid_size = sizes[len(sizes)//2]  
        
        data = results[mid_size]
        cases = list(data.keys())
        comparisons = [data[case]["avg_comparisons"] for case in cases]
        swaps = [data[case]["avg_swaps"] for case in cases]
        
        x = np.arange(len(cases))
        width = 0.35
        
        plt.bar(x - width/2, comparisons, width, label='Comparisons', color='skyblue')
        plt.bar(x + width/2, swaps, width, label='Swaps', color='salmon')
        
        plt.title(f"Operation Counts (Size {mid_size})")
        plt.xlabel("Test Cases")
        plt.ylabel("Count (log scale)")
        plt.yscale('log')
        plt.xticks(x, cases, rotation=45)
        plt.legend()

    def _plot_partition_quality(self, results):
        
        sizes = list(results.keys())
        qualities = []
        labels = []
        
        for size in sizes:
            for case, info in results[size].items():
                qualities.append(info["avg_partition_quality"])
                labels.append(f"{case}\n(size {size})")
        

        sorted_data = sorted(zip(labels, qualities), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_qualities = zip(*sorted_data)
        
        bars = plt.bar(sorted_labels[:10], sorted_qualities[:10], alpha=0.7, color='purple')
        
        plt.title("Top 10 Partition Quality Scores")
        plt.xlabel("Test Cases")
        plt.ylabel("Quality Score (0-1)")
        plt.xticks(rotation=45)
        

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', color='white')

    def _plot_scaling(self, results):
        
        sizes = list(results.keys())
        

        times = [results[size]["random"]["avg_time"] for size in sizes]
        
        plt.plot(sizes, times, marker='o', linewidth=2, color='cyan')
        

        if len(sizes) > 1:

            log_sizes = np.log(sizes)
            log_times = np.log(times)
            

            slope, intercept = np.polyfit(log_sizes, log_times, 1)
            

            fit_times = np.exp(intercept + slope * log_sizes)
            plt.plot(sizes, fit_times, 'r--', linewidth=1)
            

            complexity = "O(n)" if slope < 1.2 else "O(n log n)" if slope < 2.2 else "O(n²)" if slope < 2.8 else "O(n³)"
            plt.text(sizes[-1], times[-1], f"Complexity: {complexity}\nSlope: {slope:.2f}", 
                    ha='right', va='bottom', color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        plt.title("Scaling Behavior (Random Data)")
        plt.xlabel("Array Size")
        plt.ylabel("Time (seconds)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

    def _plot_success_rates(self, results):
        
        sizes = list(results.keys())
        largest_size = max(sizes)  
        
        data = results[largest_size]
        cases = list(data.keys())
        success_rates = [data[case]["success_rate"] * 100 for case in cases]
        
        bars = plt.bar(cases, success_rates, alpha=0.8, color='green')
        

        for i, rate in enumerate(success_rates):
            if rate < 100:
                bars[i].set_color('red')
        
        plt.title(f"Sort Success Rates (Size {largest_size})")
        plt.xlabel("Test Cases")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 105)  
        plt.xticks(rotation=45)
        

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', color='white')

    def _plot_time_distribution(self, results):
        
        sizes = list(results.keys())
        

        case_times = {}
        for size in sizes:
            for case in results[size]:
                if case not in case_times:
                    case_times[case] = []
                case_times[case].append(results[size][case]["avg_time"])
        

        plt.boxplot([times for case, times in case_times.items()], 
                   labels=case_times.keys(),
                   patch_artist=True,
                   boxprops=dict(facecolor='cyan', alpha=0.6),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
        
        plt.title("Time Distribution Across Array Sizes")
        plt.xlabel("Test Cases")
        plt.ylabel("Time (seconds)")
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)

    def _plot_efficiency(self, results):
        
        sizes = list(results.keys())
        

        efficiency = [results[size]["random"]["avg_time"] / size for size in sizes]
        
        plt.plot(sizes, efficiency, marker='s', linewidth=2, color='magenta')
        
        plt.title("Sorting Efficiency (Time per Element)")
        plt.xlabel("Array Size")
        plt.ylabel("Time per Element (seconds)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        

        for i, size in enumerate(sizes):
            plt.annotate(f"{efficiency[i]:.2e}s", 
                        (size, efficiency[i]),
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=9,
                        color='white')

    def _plot_recursion_depth(self, results):
        
        sizes = list(results.keys())
        largest_size = max(sizes)  
        
        data = results[largest_size]
        cases = list(data.keys())
        depths = [data[case]["max_recursion"] for case in cases]
        

        theoretical_depth = np.log2(largest_size)
        
        bars = plt.bar(cases, depths, alpha=0.7, color='blue')
        

        plt.axhline(y=theoretical_depth, color='r', linestyle='--', label=f'log₂(n) = {theoretical_depth:.1f}')
        
        plt.title(f"Maximum Recursion Depth (Size {largest_size})")
        plt.xlabel("Test Cases")
        plt.ylabel("Depth")
        plt.xticks(rotation=45)
        plt.legend()
        

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', color='white')


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("quantum_sort_log.txt")
        ]
    )
    
    logger = logging.getLogger("ChaosSorter")
    logger.info("Starting TheZs Quantum Sorting Algorithm Suite")
    

    sorter = TheZs()
    analyzer = TheZsAnalyzer()
    visualizer = TheZsVisualizer(analyzer)
    benchmarker = TheZsBenchmarker()
    

    logger.info("Running basic functionality test...")
    test_array = random.sample(range(1000), 1000)
    sorted_array, stats = sorter.chaos_sort(test_array)
    

    is_correct = all(sorted_array[i] <= sorted_array[i+1] for i in range(len(sorted_array)-1))
    logger.info(f"Basic sort test {'PASSED' if is_correct else 'FAILED'}")
    logger.info(f"Sorting completed in {stats.time_taken:.6f} seconds")
    logger.info(f"Comparisons: {stats.comparisons}")
    logger.info(f"Swaps: {stats.swaps}")
    

    logger.info("\nRunning comprehensive benchmark suite...")
    benchmark_results = benchmarker.run_benchmark_suite(
        sizes=[1000, 10000, 50000, 100000], 
        iterations=5
    )
    

    logger.info("\nGenerating benchmark visualizations...")
    benchmarker.visualize_benchmarks(benchmark_results)
    

    logger.info("\nRunning detailed performance analysis...")
    analysis_results = analyzer.analyze_performance(
        sizes=[100, 1000, 10000, 50000],
        repetitions=3
    )
    

    logger.info("\nGenerating performance analysis visualizations...")
    visualizer.visualize_performance(analysis_results)
    
    logger.info("\nAnalysis and benchmarking complete!")
    logger.info(f"Results saved to {benchmarker.results_file}")
    logger.info("Visualizations saved as PNG files")




