# Performance Benchmarks

This directory contains performance benchmarks for Agent-OS components.

## Running Benchmarks

### Run all benchmarks
```bash
pytest benchmarks/ --benchmark-only -v
```

### Run specific benchmark category
```bash
# Core benchmarks (parser, validator)
pytest benchmarks/ -m core --benchmark-only

# Kernel benchmarks
pytest benchmarks/ -m kernel --benchmark-only

# Agent benchmarks
pytest benchmarks/ -m agents --benchmark-only

# Memory benchmarks
pytest benchmarks/ -m memory --benchmark-only
```

### Compare with baseline
```bash
# Save current results as baseline
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline
```

### Generate JSON report
```bash
pytest benchmarks/ --benchmark-only --benchmark-json=results.json
```

### Detailed output
```bash
pytest benchmarks/ --benchmark-only \
    --benchmark-columns=min,max,mean,stddev,median,rounds \
    --benchmark-sort=mean
```

## Performance Targets

| Component | Target | Description |
|-----------|--------|-------------|
| Constitution parsing | <100ms | Parse typical constitution document |
| Rule validation | <1ms | Validate individual rule |
| Intent parsing | <10ms | Parse user intent |
| Agent routing | <100ms | Route request to appropriate agent |
| Memory operations | <100ms | Store/retrieve memory entries |
| **Total orchestration** | **<2 seconds** | Complete request flow overhead |

## Benchmark Files

- `bench_core.py` - Constitution parser and validator benchmarks
- `bench_kernel.py` - Kernel engine and policy benchmarks
- `bench_agents.py` - Agent routing and message bus benchmarks
- `bench_memory.py` - Memory vault and storage benchmarks
- `conftest.py` - Shared fixtures and configuration

## CI Integration

Benchmarks run automatically on:
- Push to main branch
- Pull requests modifying `src/` or `benchmarks/`

Results are:
- Compared against baseline (main branch)
- Posted as PR comments
- Stored as artifacts

Performance regressions >10% will fail the CI check.
