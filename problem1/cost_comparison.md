# Cost Comparison Analysis

This document outlines a credible cost comparison between self-hosting ChromaDB on a VPS and using managed services like Pinecone and Weaviate Cloud.

## Assumptions
- **Vector Dimensionality**: 384 dimensions (using `all-MiniLM-L6-v2`).
- **Storage per Vector**: 384 * 4 bytes (float32) + ~200 bytes metadata = ~1.73 KB per vector.
- **ChromaDB**:
  - Embedded / self-hosted on a VPS.
  - 100K to 1M vectors: comfortably runs on a $10/mo VPS (2 vCPU, 4 GB RAM).
  - 10M vectors: requires upgrading to a $20/mo VPS (4 vCPU, 8 GB RAM) to handle larger memory overhead and indices.
- **Pinecone**:
  - Managed `s1.x1` pod-based model.
  - Cost: $0.096/hr per pod (~$70/mo per pod).
  - 100K to 1M vectors: 1 pod is sufficient ($70/mo).
  - 10M vectors: requires 2 pods ($140/mo) because each pod holds ~5M vectors.
- **Weaviate Cloud (WCD)**:
  - Standard tier.
  - Cost: Base tier starts at $25/mo (covers up to ~1M vectors).
  - 10M vectors: standard billing scales up to ~$185/mo based on dimensions and storage units.
- **Query Volume**: 50,000 queries per month. Query embedding is computed locally (cost ~0). Managed vector databases include this search volume in their base/pod charges.

## Cost Comparison Table

| Scale | ChromaDB (self-hosted) | Pinecone (s1.x1) | Weaviate Cloud | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **100K vectors** | $10/mo | $70/mo | $25/mo | Chroma runs on basic VPS; Pinecone requires 1 pod minimum; Weaviate standard base tier. |
| **1M vectors** | $10/mo | $70/mo | $25/mo | Chroma easily fits in memory on 4GB VPS; Pinecone 1 pod; Weaviate standard base tier. |
| **10M vectors** | $20/mo (upgrade VPS) | $140/mo (2 pods) | $185/mo | Chroma upgraded to 8GB VPS ($20/mo); Pinecone requires 2 pods; Weaviate standard tier scales up. |

## Trade-offs

### When ChromaDB (Self-Hosted) Wins
- **Low-to-Medium Scale**: If your corpus contains fewer than 2-3 million vectors, you can easily host ChromaDB on a cheap virtual machine.
- **Cost Sensitivity**: Perfect for bootstrapping, internal tools, and startups where keeping baseline costs to $10/mo is vital.
- **Zero Infra Setup / No Network Overhead**: ChromaDB is fully embedded. There are no network requests, firewalls, or TLS certificates to configure to reach a database server.
- **Data Privacy**: No data leaves your secure execution environment.

### When Managed Services (Pinecone / Weaviate Cloud) Win
- **Large Scale (>10M vectors)**: When dealing with massive datasets, memory management and graph indexing become major issues that are hard to scale on a single machine.
- **High Availability & SLA**: Managed services provide multi-region replication, automated failover, backup restoration, and guaranteed uptime SLAs (e.g., 99.9%).
- **DevOps/Ops Bandwidth**: If your engineering team is small and doesn't want to spend time configuring Linux updates, Docker containers, monitoring memory leaks, or managing backups, paying the managed premium is highly worth it.
