# Enterprise Information Security & Compliance Policy (SEC-2026-v4)

## 1. Authentication and Identity Management
All corporate workforce members and autonomous agent service principals must authenticate via FIDO2 WebAuthn hardware security keys or Phishing-Resistant Multi-Factor Authentication (MFA). Legacy SMS or time-based one-time password (TOTP) mechanisms are strictly prohibited for production environments.

Single Sign-On (SSO) sessions enforce a mandatory 8-hour maximum lifetime. Re-authentication is required immediately upon detecting an IP anomaly or device posture degradation via the Zero Trust Network Access (ZTNA) policy engine. Service account tokens must be rotated every 90 days, or every 24 hours if classified as high-privilege tier-0 credentials.

## 2. Encryption and Key Management
Data at rest across all relational databases, object storage buckets (Amazon S3, Google Cloud Storage), and distributed vector stores must be encrypted using AES-256-GCM or ChaCha20-Poly1305. Customer Managed Encryption Keys (CMEK) via AWS KMS or GCP Cloud KMS are required for all PII and financial records. Key rotation must occur automatically on a 365-day schedule.

Data in transit requires TLS 1.3 with mandatory forward secrecy (ECDHE-ECDSA-AES256-GCM-SHA384 or ECDHE-RSA-AES256-GCM-SHA384). TLS 1.0, 1.1, and 1.2 are deprecated and actively blocked by boundary Application Load Balancers.

## 3. Autonomous AI Agent and LLM Security Controls
All LLM prompts and retrieved vector contexts must pass through the Aegis Content Firewall prior to model execution. The firewall scans for:
- Indirect Prompt Injections (IPI) using heuristic perplexity scoring and structural delimiters.
- Personally Identifiable Information (PII) redaction with Presidio NER models (Social Security numbers, credit card numbers, HIPAA indicators).
- Cross-Tenant context leakage: Vectors must be strictly isolated using tenant_id metadata filters in ChromaDB/Qdrant.

Vector similarity thresholds alone must never be used for security-critical access decisions. All queries must enforce role-based access control (RBAC) at the document ingestion and retrieval layer.

## 4. Incident Response and Breach Notification SLA
In the event of a confirmed security breach involving unencrypted customer data:
- The Computer Security Incident Response Team (CSIRT) must be convened within 15 minutes of P1 alarm triggering.
- Regulatory bodies (GDPR, SEC, CCPA) must be notified within 72 hours of confirmation.
- Affected enterprise customers must receive notification within 24 hours of technical confirmation.

