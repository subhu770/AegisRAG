# AegisRAG Cloud Infrastructure & Kubernetes Architecture Specification

## 1. System Topology & Pod Sizing
The AegisRAG inference service is deployed on Kubernetes (EKS / GKE) across three Availability Zones for high availability. 

### Worker Pod Resource Allocations:
- CPU: 4 vCPU requests, 8 vCPU limits (AMD EPYC or Intel Xeon Platinum).
- Memory: 8Gi requests, 16Gi limits to accommodate cold-start SentenceTransformer weights and BM25 inverted indexes.
- Accelerators: Optional NVIDIA T4 or L4 GPU for low-latency Cross-Encoder inference (batch latency < 15ms).
- Horizontal Pod Autoscaling (HPA): Triggers when average CPU utilization exceeds 70% or P95 request latency exceeds 250ms. Minimum replica count is 3; maximum replica count is 30.

## 2. Ingress and Traffic Management
All ingress traffic is terminated at an Envoy-based API Gateway supporting HTTP/2 and gRPC:
- Rate Limiting: 1,000 requests per minute per enterprise tenant API key with token-bucket algorithm.
- Circuit Breaking: Configured with 50 consecutive 5xx errors triggering a 30-second circuit open state.
- Timeout Budget: Global request deadline is capped at 5,000ms. Retrieval stage budget: 350ms; Reranking stage budget: 600ms; LLM generation budget: 3,500ms.

## 3. Storage Subsystems
- Vector Store Persistence: ChromaDB persistent state is mounted to high-throughput AWS gp3 EBS volumes provisioned with 3,000 IOPS and 125 MB/s baseline throughput.
- Read Replicas: For enterprise scale beyond 10 million chunks, the system federates read queries across read-only vector replicas with asynchronous write journaling.
- Backup Policy: Hourly snapshotting of EBS volumes with 30-day retention; BM25 tokenized inverted index state is dumped to encrypted S3 buckets upon index modifications.

## 4. Disaster Recovery & Recovery Point Objective
- Recovery Time Objective (RTO): < 30 minutes for complete multi-region failover.
- Recovery Point Objective (RPO): < 5 minutes of indexed document delta.

