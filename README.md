# Vector Database

A high-performance vector database implementation with multiple backends, distributed replication, and sharding support. This project provides a FastAPI-based REST API for similarity search operations on high-dimensional vectors.

## Overview

This vector database supports multiple storage backends optimized for different use cases:
- **Brute Force**: Exact similarity search using linear scan
- **HNSW**: Approximate nearest neighbor search using Hierarchical Navigable Small World graphs
- **Sharded HNSW**: Parallel query processing across multiple HNSW shards
- **Distributed**: Multi-replica setup with write-ahead logging and intelligent read routing

## Core Components

### `app.py`
FastAPI wrapper that provides REST API endpoints for the vector database. Exposes endpoints for:
- Creating collections (`POST /collections`)
- Upserting vectors (`POST /upsert`)
- Deleting vectors (`POST /delete`)
- Querying for similar vectors (`POST /query`)

### `store.py`
Contains the core vector database implementations:
- **`BaseCollection`**: Abstract base class defining the collection interface
- **`BruteCollection`**: Brute force implementation that performs exact similarity search by computing distances to all vectors. Supports cosine similarity and L2 distance metrics.
- **`HNSWCollection`**: HNSW-based implementation using the `hnswlib` library for approximate nearest neighbor search. Provides faster query times at the cost of approximate results.

### `distributed_collection.py`
Implements a distributed vector database with replication:
- **`WalDistributedCollection`**: Distributed collection that maintains:
  - A primary replica that applies writes immediately
  - Multiple follower replicas that apply writes asynchronously via WAL
  - Intelligent read routing that selects the best up-to-date replica based on latency and load
  - Support for synchronous or asynchronous replication modes
  - Configurable artificial network delays for testing

### `wal.py`
Write-Ahead Log (WAL) implementation for replication:
- **`WriteAheadLog`**: Thread-safe log that records all write operations (upsert/delete)
- Tracks Log Sequence Numbers (LSNs) for each operation
- Supports replica acknowledgment and replication tracking
- Enables consistency guarantees by ensuring replicas are up-to-date before serving reads

### `replica_worker.py`
Replica server implementation:
- **`ReplicaWorker`**: Background thread that continuously applies WAL entries to a replica collection
- Maintains its own copy of the collection (using any backend: brute, hnsw, etc.)
- Reports progress back to the WAL for consistency tracking
- Includes configurable replication lag simulation

### `sharded_hnsw.py`
Sharded HNSW implementation for parallel query processing:
- **`ShardedHNSWCollection`**: Distributes vectors across multiple HNSW shards using round-robin assignment
- Queries are executed in parallel across shards using multiprocessing
- Results from all shards are merged to return the global top-k results
- Automatically manages process pools and handles platform differences (fork vs spawn)

### `vectordb_types.py`
Pydantic models defining the API request/response schemas:
- `CreateCollection`: Collection creation parameters
- `Point`: Vector point with id, vector, and optional metadata
- `UpsertRequest`, `DeleteRequest`, `QueryRequest`: API request types
- `QueryHit`, `QueryResponse`: Query result types

## Benchmarking and Testing

### Benchmark Scripts
- **`benchmark_brute_vs_hnsw.py`**: Compares brute force vs HNSW performance (recall, latency, scaling)
- **`bench_latency_percentiles.py`**: Measures latency percentiles for distributed collections
- **`bench_lag_consistency.py`**: Tests replication lag and consistency guarantees
- **`bench_routing_histogram.py`**: Analyzes read routing behavior across replicas

### Test Scripts
- **`test_hnsw.py`**: Unit tests for HNSW collection correctness
- **`test_wal_distributed.py`**: Tests for distributed collection and WAL functionality

## AI Usage
For this project, we utilized AI to help with debugging, syntax, and building a general framework for our implementation. AI helped with making the types, API formatting, and the BaseCollection in store.py. Furthermore, we also asked AI for help in coming up with experiments to run for our project. 

## Usage

### Starting the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Creating a Collection

```bash
curl -X POST "http://localhost:8000/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_collection",
    "dim": 384,
    "metric": "cosine",
    "backend": "hnsw"
  }'
```

### Upserting Vectors

```bash
curl -X POST "http://localhost:8000/upsert?collection=my_collection" \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "vec1",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"category": "example"}
      }
    ]
  }'
```

### Querying

```bash
curl -X POST "http://localhost:8000/query?collection=my_collection" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "top_k": 10
  }'
```

### Deleting Vectors

```bash
curl -X POST "http://localhost:8000/delete?collection=my_collection" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["vec1", "vec2"]
  }'
```

## Backend Types

When creating a collection, you can specify different backends:

- **`brute`**: Exact search, slower but 100% recall
- **`hnsw`**: Fast approximate search with configurable accuracy
- **`sharded_hnsw`**: Parallel HNSW queries across multiple shards
- **`distributed`**: Multi-replica setup with WAL-based replication

## Features

- **Multiple Similarity Metrics**: Supports cosine similarity and L2 distance
- **Metadata Support**: Attach arbitrary metadata to vectors
- **Distributed Replication**: Multi-replica setup with consistency guarantees
- **Intelligent Routing**: Load and latency-aware read routing
- **Sharding**: Parallel query processing across shards
- **Write-Ahead Logging**: Durable replication with LSN tracking

## Dependencies

- `fastapi`: Web framework
- `numpy`: Numerical operations
- `hnswlib`: HNSW index implementation
- `pydantic`: Data validation
- `sentence-transformers`: For embedding generation (ingestion script)

## Project Structure

```
vectordb/
├── app.py                    # FastAPI REST API
├── store.py                  # Core collection implementations
├── distributed_collection.py # Distributed/replicated collections
├── wal.py                    # Write-ahead log
├── replica_worker.py         # Replica server
├── sharded_hnsw.py           # Sharded HNSW implementation
├── vectordb_types.py         # API type definitions
├── benchmark_*.py            # Performance benchmarks
├── test_*.py                 # Test suites
└── README.md                 # This file
```

