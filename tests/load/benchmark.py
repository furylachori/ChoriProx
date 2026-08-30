#!/usr/bin/env python3
"""Quick benchmark runner for ChoriProx.

Usage:
    PYTHONPATH=src python tests/load/benchmark.py [num_requests] [concurrency]

Examples:
    PYTHONPATH=src python tests/load/benchmark.py              # 100 requests, 10 concurrent
    PYTHONPATH=src python tests/load/benchmark.py 500 50       # 500 requests, 50 concurrent
    PYTHONPATH=src python tests/load/benchmark.py 1000 100     # 1000 requests, 100 concurrent
"""
import sys
import time
import concurrent.futures
import statistics
from unittest.mock import MagicMock, patch


def setup_mocks():
    """Setup all necessary mocks for benchmarking."""
    mock_runner = MagicMock()
    mock_runner.run_pending.return_value = []
    
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock response"),
            finish_reason="stop",
        )
    ]
    mock_chat_response.model = "gpt-4"
    mock_chat_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_chat_response

    mock_openai_client = MagicMock()
    mock_openai_client.chat = MagicMock()
    mock_openai_client.chat.completions = mock_completions

    mock_sota_client = MagicMock()
    mock_sota_response = MagicMock()
    mock_sota_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock sota response"),
            finish_reason="stop",
        )
    ]
    mock_sota_response.model = "sota-model"
    mock_sota_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    mock_sota_client.chat = MagicMock()
    mock_sota_client.chat.completions = MagicMock()
    mock_sota_client.chat.completions.create.return_value = mock_sota_response

    return {
        'migration': patch('memory.migrations.runner.MigrationRunner', return_value=mock_runner),
        'openai': patch("openai.OpenAI", return_value=mock_openai_client),
        'openai_completions': patch("openai.resources.chat.Completions", return_value=mock_completions),
        'sync': patch("memory.sync.submit_sync"),
        'sota': patch("memory.sota_client.SOTAClient", return_value=mock_sota_client),
    }


def benchmark_chat(num_requests=100, concurrency=10):
    """Run benchmark against chat completions endpoint."""
    from qwen_proxy import _HTTP_SESSION, app
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Mock response"},
                "finish_reason": "stop",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    latencies = []
    errors = 0
    status_codes = {}
    
    def make_request():
        with app.test_client() as client:
            with patch.object(_HTTP_SESSION, "request", return_value=mock_response):
                start = time.time()
                try:
                    response = client.post('/v1/chat/completions', json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "benchmark test"}],
                        "stream": False,
                    })
                    elapsed = time.time() - start
                    return response.status_code, elapsed
                except Exception as e:
                    elapsed = time.time() - start
                    return 500, elapsed
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: POST /v1/chat/completions")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            status, latency = f.result()
            latencies.append(latency)
            status_codes[status] = status_codes.get(status, 0) + 1
            if status >= 400 and status != 429:
                errors += 1
    
    total_time = time.time() - start_time
    latencies.sort()
    
    print(f"RESULTS:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Throughput:     {num_requests/total_time:.1f} req/s")
    print(f"  Errors:         {errors}/{num_requests} ({errors/num_requests*100:.1f}%)")
    print(f"  Status codes:   {status_codes}")
    print(f"\nLATENCY DISTRIBUTION:")
    print(f"  Min:            {latencies[0]:.4f}s")
    print(f"  p50 (median):   {latencies[len(latencies)//2]:.4f}s")
    print(f"  p95:            {latencies[int(len(latencies)*0.95)]:.4f}s")
    print(f"  p99:            {latencies[int(len(latencies)*0.99)]:.4f}s")
    print(f"  Max:            {latencies[-1]:.4f}s")
    print(f"  Mean:           {statistics.mean(latencies):.4f}s")
    print(f"  Stdev:          {statistics.stdev(latencies):.4f}s" if len(latencies) > 1 else "")
    print(f"{'='*70}\n")


def benchmark_health(num_requests=100, concurrency=10):
    """Run benchmark against health check endpoint."""
    from qwen_proxy import app
    
    latencies = []
    errors = 0
    status_codes = {}
    
    def make_request():
        with app.test_client() as client:
            start = time.time()
            try:
                response = client.get('/health')
                elapsed = time.time() - start
                return response.status_code, elapsed
            except Exception as e:
                elapsed = time.time() - start
                return 500, elapsed
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: GET /health")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            status, latency = f.result()
            latencies.append(latency)
            status_codes[status] = status_codes.get(status, 0) + 1
            if status >= 400:
                errors += 1
    
    total_time = time.time() - start_time
    latencies.sort()
    
    print(f"RESULTS:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Throughput:     {num_requests/total_time:.1f} req/s")
    print(f"  Errors:         {errors}/{num_requests} ({errors/num_requests*100:.1f}%)")
    print(f"  Status codes:   {status_codes}")
    print(f"\nLATENCY DISTRIBUTION:")
    print(f"  Min:            {latencies[0]:.4f}s")
    print(f"  p50 (median):   {latencies[len(latencies)//2]:.4f}s")
    print(f"  p95:            {latencies[int(len(latencies)*0.95)]:.4f}s")
    print(f"  p99:            {latencies[int(len(latencies)*0.99)]:.4f}s")
    print(f"  Max:            {latencies[-1]:.4f}s")
    print(f"  Mean:           {statistics.mean(latencies):.4f}s")
    print(f"  Stdev:          {statistics.stdev(latencies):.4f}s" if len(latencies) > 1 else "")
    print(f"{'='*70}\n")


def benchmark_models(num_requests=100, concurrency=10):
    """Run benchmark against models list endpoint."""
    from qwen_proxy import app
    
    latencies = []
    errors = 0
    status_codes = {}
    
    def make_request():
        with app.test_client() as client:
            start = time.time()
            try:
                response = client.get('/v1/models')
                elapsed = time.time() - start
                return response.status_code, elapsed
            except Exception as e:
                elapsed = time.time() - start
                return 500, elapsed
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: GET /v1/models")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            status, latency = f.result()
            latencies.append(latency)
            status_codes[status] = status_codes.get(status, 0) + 1
            if status >= 400:
                errors += 1
    
    total_time = time.time() - start_time
    latencies.sort()
    
    print(f"RESULTS:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Throughput:     {num_requests/total_time:.1f} req/s")
    print(f"  Errors:         {errors}/{num_requests} ({errors/num_requests*100:.1f}%)")
    print(f"  Status codes:   {status_codes}")
    print(f"\nLATENCY DISTRIBUTION:")
    print(f"  Min:            {latencies[0]:.4f}s")
    print(f"  p50 (median):   {latencies[len(latencies)//2]:.4f}s")
    print(f"  p95:            {latencies[int(len(latencies)*0.95)]:.4f}s")
    print(f"  p99:            {latencies[int(len(latencies)*0.99)]:.4f}s")
    print(f"  Max:            {latencies[-1]:.4f}s")
    print(f"  Mean:           {statistics.mean(latencies):.4f}s")
    print(f"  Stdev:          {statistics.stdev(latencies):.4f}s" if len(latencies) > 1 else "")
    print(f"{'='*70}\n")


def main():
    """Run all benchmarks."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("\n" + "="*70)
    print("ChoriProx Load Benchmark")
    print("="*70)
    print(f"Configuration: {n} requests, {c} concurrent workers")
    print("="*70)
    
    mocks = setup_mocks()
    
    with mocks['migration'], mocks['openai'], mocks['openai_completions'], \
         mocks['sync'], mocks['sota']:
        
        benchmark_chat(n, c)
        benchmark_health(n, c)
        benchmark_models(n, c)
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
