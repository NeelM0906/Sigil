# Sigil v2 - Phase 7 Deployment Guide

**Version:** 2.0.0
**Date:** 2026-01-11
**Status:** Integration & Polish Phase

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Local Development](#3-local-development)
4. [Docker Deployment](#4-docker-deployment)
5. [Production Deployment](#5-production-deployment)
6. [Configuration Reference](#6-configuration-reference)
7. [Database Setup](#7-database-setup)
8. [Service Dependencies](#8-service-dependencies)
9. [Monitoring & Observability](#9-monitoring--observability)
10. [Security Configuration](#10-security-configuration)
11. [Scaling Guide](#11-scaling-guide)
12. [Troubleshooting](#12-troubleshooting)
13. [Maintenance Operations](#13-maintenance-operations)
14. [Disaster Recovery](#14-disaster-recovery)
15. [Appendices](#15-appendices)

---

## 1. Prerequisites

### 1.1 System Requirements

**Minimum Requirements (Development):**
| Resource | Requirement |
|----------|-------------|
| CPU | 2 cores |
| RAM | 4 GB |
| Storage | 10 GB SSD |
| Python | 3.12+ |
| OS | macOS 12+, Ubuntu 22.04+, Windows 11 (WSL2) |

**Recommended Requirements (Production):**
| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores |
| RAM | 16+ GB |
| Storage | 100+ GB SSD |
| Python | 3.12+ |
| OS | Ubuntu 22.04 LTS |

### 1.2 Required Software

```bash
# Python 3.12+
python3 --version  # Should be 3.12.x or higher

# pip (latest)
pip3 --version

# Git
git --version

# Node.js (for MCP tools)
node --version  # v18.x or higher recommended

# Docker (for containerized deployment)
docker --version
docker-compose --version
```

### 1.3 Required API Keys

| Service | Environment Variable | Required | Purpose |
|---------|---------------------|----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` | Yes | LLM provider |
| Tavily | `TAVILY_API_KEY` | No | Web search tool |
| ElevenLabs | `ELEVENLABS_API_KEY` | No | Voice synthesis |
| Twilio | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | No | SMS/Voice calls |
| Google Calendar | `GOOGLE_CALENDAR_CREDENTIALS` | No | Calendar integration |
| HubSpot | `HUBSPOT_API_KEY` | No | CRM integration |

### 1.4 Network Requirements

| Port | Service | Protocol | Direction |
|------|---------|----------|-----------|
| 8000 | FastAPI | HTTP/HTTPS | Inbound |
| 8001 | WebSocket | WS/WSS | Inbound |
| 443 | Anthropic API | HTTPS | Outbound |
| 443 | MCP Tools | HTTPS | Outbound |

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/acti-agent-builder.git
cd acti-agent-builder

# Checkout development branch
git checkout neel_dev
```

### 2.2 Virtual Environment Setup

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip
```

### 2.3 Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (if developing)
pip install -r requirements-dev.txt

# Install MCP tools (optional)
npm install -g @anthropic/mcp
```

### 2.4 Environment Variables

Create a `.env` file in the project root:

```bash
# =============================================================================
# SIGIL V2 ENVIRONMENT CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Core API Keys (Required)
# -----------------------------------------------------------------------------
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# -----------------------------------------------------------------------------
# Feature Flags
# -----------------------------------------------------------------------------
SIGIL_USE_MEMORY=false
SIGIL_USE_PLANNING=false
SIGIL_USE_CONTRACTS=false
SIGIL_USE_EVOLUTION=false
SIGIL_USE_ROUTING=true

# -----------------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------------
SIGIL_LLM_PROVIDER=anthropic
SIGIL_LLM_MODEL=claude-sonnet-4-20250514
SIGIL_LLM_TEMPERATURE=0.7
SIGIL_LLM_MAX_TOKENS=4096
SIGIL_LLM_TIMEOUT=60

# -----------------------------------------------------------------------------
# Memory Configuration
# -----------------------------------------------------------------------------
SIGIL_MEMORY_EMBEDDING_MODEL=text-embedding-3-small
SIGIL_MEMORY_RETRIEVAL_K=5
SIGIL_MEMORY_HYBRID_THRESHOLD=0.7
SIGIL_MEMORY_DATA_DIR=./data/memory

# -----------------------------------------------------------------------------
# Contract Configuration
# -----------------------------------------------------------------------------
SIGIL_CONTRACT_DEFAULT_MAX_RETRIES=2
SIGIL_CONTRACT_DEFAULT_STRATEGY=retry
SIGIL_CONTRACT_MAX_TOTAL_TOKENS=256000

# -----------------------------------------------------------------------------
# State Configuration
# -----------------------------------------------------------------------------
SIGIL_STATE_DATA_DIR=./data/state
SIGIL_STATE_MAX_EVENTS_PER_SESSION=10000

# -----------------------------------------------------------------------------
# API Server Configuration
# -----------------------------------------------------------------------------
SIGIL_API_HOST=0.0.0.0
SIGIL_API_PORT=8000
SIGIL_API_WORKERS=4
SIGIL_API_DEBUG=false
SIGIL_API_CORS_ORIGINS=["http://localhost:3000"]

# -----------------------------------------------------------------------------
# Security Configuration
# -----------------------------------------------------------------------------
SIGIL_JWT_SECRET=your-jwt-secret-key-here-min-32-chars
SIGIL_JWT_ALGORITHM=HS256
SIGIL_JWT_EXPIRE_MINUTES=60
SIGIL_RATE_LIMIT_PER_MINUTE=60

# -----------------------------------------------------------------------------
# MCP Tool Configuration (Optional)
# -----------------------------------------------------------------------------
TAVILY_API_KEY=tvly-your-key-here
ELEVENLABS_API_KEY=your-eleven-labs-key
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
SIGIL_LOG_LEVEL=INFO
SIGIL_LOG_FORMAT=json
SIGIL_LOG_FILE=./logs/sigil.log

# -----------------------------------------------------------------------------
# Telemetry Configuration
# -----------------------------------------------------------------------------
SIGIL_TELEMETRY_ENABLED=true
SIGIL_TELEMETRY_ENDPOINT=
```

### 2.5 Verify Installation

```bash
# Verify Python environment
python -c "import sigil; print(f'Sigil v{sigil.__version__}')"

# Run quick health check
python -m sigil.health_check

# Run tests
pytest tests/ -v --tb=short
```

---

## 3. Local Development

### 3.1 Development Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start development server (when implemented)
python -m sigil.interfaces.api --reload --debug

# Alternative: Run CLI
python -m sigil.interfaces.cli
```

### 3.2 Development Configuration

Create `config/development.yaml`:

```yaml
# Development-specific configuration
sigil:
  environment: development
  debug: true

  llm:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096

  features:
    memory: true
    planning: true
    contracts: true
    evolution: false
    routing: true

  memory:
    data_dir: ./data/dev/memory
    embedding_model: text-embedding-3-small

  state:
    data_dir: ./data/dev/state

  api:
    host: 127.0.0.1
    port: 8000
    debug: true
    reload: true

  logging:
    level: DEBUG
    format: pretty
```

### 3.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/memory/ -v
pytest tests/planning/ -v
pytest tests/reasoning/ -v
pytest tests/contracts/ -v

# Run with coverage
pytest tests/ --cov=sigil --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"

# Run integration tests
pytest tests/integration/ -v
```

### 3.4 Code Quality

```bash
# Run linters
ruff check sigil/
mypy sigil/

# Format code
ruff format sigil/

# Run pre-commit hooks
pre-commit run --all-files
```

### 3.5 Development Workflow

```
1. Create feature branch
   git checkout -b feature/your-feature

2. Make changes
   - Write code
   - Add tests
   - Update documentation

3. Run quality checks
   pytest tests/ -v
   ruff check sigil/
   mypy sigil/

4. Commit with descriptive message
   git add .
   git commit -m "feat: Add your feature"

5. Push and create PR
   git push origin feature/your-feature
```

---

## 4. Docker Deployment

### 4.1 Dockerfile

Create `Dockerfile` in project root:

```dockerfile
# =============================================================================
# SIGIL V2 DOCKERFILE
# =============================================================================

# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash sigil

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application code
COPY sigil/ ./sigil/
COPY config/ ./config/

# Create data directories
RUN mkdir -p /app/data/memory /app/data/state /app/logs
RUN chown -R sigil:sigil /app

# Switch to non-root user
USER sigil

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SIGIL_MEMORY_DATA_DIR=/app/data/memory
ENV SIGIL_STATE_DATA_DIR=/app/data/state
ENV SIGIL_LOG_FILE=/app/logs/sigil.log

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "sigil.interfaces.api", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Docker Compose

Create `docker-compose.yml`:

```yaml
# =============================================================================
# SIGIL V2 DOCKER COMPOSE
# =============================================================================

version: '3.8'

services:
  # ---------------------------------------------------------------------------
  # Sigil API Server
  # ---------------------------------------------------------------------------
  sigil-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: sigil-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SIGIL_USE_MEMORY=${SIGIL_USE_MEMORY:-true}
      - SIGIL_USE_PLANNING=${SIGIL_USE_PLANNING:-true}
      - SIGIL_USE_CONTRACTS=${SIGIL_USE_CONTRACTS:-true}
      - SIGIL_JWT_SECRET=${SIGIL_JWT_SECRET}
      - SIGIL_API_DEBUG=false
    volumes:
      - sigil-memory:/app/data/memory
      - sigil-state:/app/data/state
      - sigil-logs:/app/logs
    networks:
      - sigil-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # ---------------------------------------------------------------------------
  # Redis Cache (Optional - for scaling)
  # ---------------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: sigil-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - sigil-redis:/data
    networks:
      - sigil-network
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ---------------------------------------------------------------------------
  # Vector Database (Optional - for production memory)
  # ---------------------------------------------------------------------------
  qdrant:
    image: qdrant/qdrant:latest
    container_name: sigil-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - sigil-qdrant:/qdrant/storage
    networks:
      - sigil-network
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  # ---------------------------------------------------------------------------
  # PostgreSQL (Optional - for production state)
  # ---------------------------------------------------------------------------
  postgres:
    image: postgres:15-alpine
    container_name: sigil-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=sigil
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-sigil_secure_password}
      - POSTGRES_DB=sigil
    volumes:
      - sigil-postgres:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - sigil-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sigil"]
      interval: 10s
      timeout: 5s
      retries: 5

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
networks:
  sigil-network:
    driver: bridge

# -----------------------------------------------------------------------------
# Volumes
# -----------------------------------------------------------------------------
volumes:
  sigil-memory:
  sigil-state:
  sigil-logs:
  sigil-redis:
  sigil-qdrant:
  sigil-postgres:
```

### 4.3 Build and Run

```bash
# Build the image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f sigil-api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### 4.4 Docker Development Mode

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  sigil-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./sigil:/app/sigil:ro
      - ./tests:/app/tests:ro
    environment:
      - SIGIL_API_DEBUG=true
      - SIGIL_LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"
      - "5678:5678"  # Debug port
```

---

## 5. Production Deployment

### 5.1 Production Checklist

**Pre-Deployment:**
- [ ] All tests passing
- [ ] Security audit completed
- [ ] Performance testing completed
- [ ] API documentation updated
- [ ] Monitoring configured
- [ ] Backup strategy defined
- [ ] Rollback plan documented

**Configuration:**
- [ ] Production environment variables set
- [ ] Debug mode disabled
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] JWT secrets rotated
- [ ] API keys verified

**Infrastructure:**
- [ ] SSL/TLS certificates configured
- [ ] Load balancer configured
- [ ] Health checks configured
- [ ] Auto-scaling policies set
- [ ] Database backups scheduled

### 5.2 Production Environment Variables

```bash
# =============================================================================
# PRODUCTION ENVIRONMENT
# =============================================================================

# Core
ANTHROPIC_API_KEY=sk-ant-api03-production-key
SIGIL_ENVIRONMENT=production

# Feature Flags (enable all for production)
SIGIL_USE_MEMORY=true
SIGIL_USE_PLANNING=true
SIGIL_USE_CONTRACTS=true
SIGIL_USE_EVOLUTION=false  # Enable after testing
SIGIL_USE_ROUTING=true

# API Configuration
SIGIL_API_HOST=0.0.0.0
SIGIL_API_PORT=8000
SIGIL_API_WORKERS=4
SIGIL_API_DEBUG=false

# Security (REQUIRED - Generate new secrets!)
SIGIL_JWT_SECRET=<generate-64-char-random-string>
SIGIL_JWT_ALGORITHM=HS256
SIGIL_JWT_EXPIRE_MINUTES=30
SIGIL_RATE_LIMIT_PER_MINUTE=100

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://sigil:password@postgres:5432/sigil

# Redis (if using for caching)
REDIS_URL=redis://redis:6379/0

# Vector Database (if using Qdrant)
QDRANT_URL=http://qdrant:6333

# Logging
SIGIL_LOG_LEVEL=INFO
SIGIL_LOG_FORMAT=json

# Monitoring
SIGIL_TELEMETRY_ENABLED=true
SIGIL_TELEMETRY_ENDPOINT=https://your-monitoring-service.com/ingest
```

### 5.3 Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
# =============================================================================
# SIGIL V2 NGINX CONFIGURATION
# =============================================================================

upstream sigil_api {
    server sigil-api:8000;
    keepalive 32;
}

upstream sigil_ws {
    server sigil-api:8001;
    keepalive 32;
}

server {
    listen 80;
    server_name api.sigil.io;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.sigil.io;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/sigil.crt;
    ssl_certificate_key /etc/nginx/ssl/sigil.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    # API Endpoints
    location /api/ {
        proxy_pass http://sigil_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket Endpoints
    location /ws/ {
        proxy_pass http://sigil_ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    # Health Check (no auth required)
    location /health {
        proxy_pass http://sigil_api/health;
        proxy_http_version 1.1;
        access_log off;
    }

    # Metrics (internal only)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://sigil_api/metrics;
    }

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript;
}
```

### 5.4 Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
# =============================================================================
# SIGIL V2 KUBERNETES DEPLOYMENT
# =============================================================================

apiVersion: apps/v1
kind: Deployment
metadata:
  name: sigil-api
  namespace: sigil
  labels:
    app: sigil-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sigil-api
  template:
    metadata:
      labels:
        app: sigil-api
    spec:
      containers:
        - name: sigil-api
          image: your-registry/sigil:v2.0.0
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: sigil-secrets
            - configMapRef:
                name: sigil-config
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /app/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: sigil-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: sigil-api
  namespace: sigil
spec:
  selector:
    app: sigil-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sigil-api-hpa
  namespace: sigil
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sigil-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 5.5 Cloud Provider Guides

**AWS (ECS/Fargate):**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name sigil-cluster

# Push image to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
docker tag sigil:latest $ECR_REGISTRY/sigil:latest
docker push $ECR_REGISTRY/sigil:latest

# Create service
aws ecs create-service \
  --cluster sigil-cluster \
  --service-name sigil-api \
  --task-definition sigil-api:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

**GCP (Cloud Run):**
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/$PROJECT_ID/sigil

# Deploy to Cloud Run
gcloud run deploy sigil-api \
  --image gcr.io/$PROJECT_ID/sigil \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10
```

**Azure (Container Apps):**
```bash
# Create container app environment
az containerapp env create \
  --name sigil-env \
  --resource-group sigil-rg

# Deploy container app
az containerapp create \
  --name sigil-api \
  --resource-group sigil-rg \
  --environment sigil-env \
  --image sigilregistry.azurecr.io/sigil:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10
```

---

## 6. Configuration Reference

### 6.1 Environment Variables Reference

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `ANTHROPIC_API_KEY` | - | Anthropic API key | Yes |
| `SIGIL_USE_MEMORY` | false | Enable memory system | No |
| `SIGIL_USE_PLANNING` | false | Enable planning system | No |
| `SIGIL_USE_CONTRACTS` | false | Enable contracts system | No |
| `SIGIL_USE_EVOLUTION` | false | Enable evolution system | No |
| `SIGIL_USE_ROUTING` | true | Enable routing system | No |
| `SIGIL_LLM_PROVIDER` | anthropic | LLM provider name | No |
| `SIGIL_LLM_MODEL` | claude-sonnet-4-20250514 | Model identifier | No |
| `SIGIL_LLM_TEMPERATURE` | 0.7 | Generation temperature | No |
| `SIGIL_LLM_MAX_TOKENS` | 4096 | Max output tokens | No |
| `SIGIL_API_HOST` | 0.0.0.0 | API bind host | No |
| `SIGIL_API_PORT` | 8000 | API bind port | No |
| `SIGIL_API_WORKERS` | 1 | Number of workers | No |
| `SIGIL_JWT_SECRET` | - | JWT signing secret | Yes (prod) |
| `SIGIL_LOG_LEVEL` | INFO | Logging level | No |

### 6.2 Settings Classes

```python
# sigil/config/settings.py

class SigilSettings(BaseSettings):
    """Main configuration class."""

    # Feature flags
    use_memory: bool = False
    use_planning: bool = False
    use_contracts: bool = False
    use_evolution: bool = False
    use_routing: bool = True

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    contracts: ContractSettings = Field(default_factory=ContractSettings)

    class Config:
        env_prefix = "SIGIL_"
        env_file = ".env"
        env_nested_delimiter = "__"


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class MemorySettings(BaseSettings):
    """Memory system settings."""
    data_dir: str = "./data/memory"
    embedding_model: str = "text-embedding-3-small"
    retrieval_k: int = 5
    hybrid_threshold: float = 0.7


class ContractSettings(BaseSettings):
    """Contract system settings."""
    default_max_retries: int = 2
    default_strategy: str = "retry"
    max_total_tokens: int = 256000
```

### 6.3 Configuration Precedence

```
1. Command line arguments (highest priority)
2. Environment variables
3. .env file
4. config/*.yaml files
5. Default values (lowest priority)
```

---

## 7. Database Setup

### 7.1 PostgreSQL Schema

Create `scripts/init-db.sql`:

```sql
-- =============================================================================
-- SIGIL V2 DATABASE SCHEMA
-- =============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------------------------
-- Events Table (Event Sourcing)
-- -----------------------------------------------------------------------------
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id VARCHAR(100) NOT NULL,
    agent_id VARCHAR(100),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_session ON events(session_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(event_type);

-- -----------------------------------------------------------------------------
-- Resources Table (Memory Layer 1)
-- -----------------------------------------------------------------------------
CREATE TABLE resources (
    resource_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    session_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_resources_type ON resources(resource_type);
CREATE INDEX idx_resources_session ON resources(session_id);

-- -----------------------------------------------------------------------------
-- Items Table (Memory Layer 2)
-- -----------------------------------------------------------------------------
CREATE TABLE memory_items (
    item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_resource_id UUID REFERENCES resources(resource_id),
    content TEXT NOT NULL,
    category VARCHAR(100),
    confidence FLOAT DEFAULT 1.0,
    embedding VECTOR(1536),  -- Requires pgvector extension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_items_category ON memory_items(category);
CREATE INDEX idx_items_source ON memory_items(source_resource_id);

-- -----------------------------------------------------------------------------
-- Categories Table (Memory Layer 3)
-- -----------------------------------------------------------------------------
CREATE TABLE categories (
    category_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    item_count INTEGER DEFAULT 0,
    last_consolidated TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_categories_name ON categories(name);

-- -----------------------------------------------------------------------------
-- Contracts Table
-- -----------------------------------------------------------------------------
CREATE TABLE contracts (
    contract_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    specification JSONB NOT NULL,
    version VARCHAR(20) DEFAULT '1.0.0',
    is_template BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_contracts_name ON contracts(name);
CREATE INDEX idx_contracts_template ON contracts(is_template);

-- -----------------------------------------------------------------------------
-- Sessions Table
-- -----------------------------------------------------------------------------
CREATE TABLE sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    agent_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE INDEX idx_sessions_status ON sessions(status);

-- -----------------------------------------------------------------------------
-- API Keys Table (for authentication)
-- -----------------------------------------------------------------------------
CREATE TABLE api_keys (
    key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(64) NOT NULL,
    name VARCHAR(100) NOT NULL,
    scopes JSONB DEFAULT '[]',
    rate_limit INTEGER DEFAULT 60,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX idx_api_keys_hash ON api_keys(key_hash);
```

### 7.2 Database Migrations

Using Alembic for migrations:

```bash
# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

### 7.3 Vector Database (Qdrant)

```python
# Example Qdrant collection setup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create collection for memory items
client.create_collection(
    collection_name="memory_items",
    vectors_config=VectorParams(
        size=1536,  # OpenAI embedding dimension
        distance=Distance.COSINE
    )
)

# Create collection for categories
client.create_collection(
    collection_name="categories",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
```

---

## 8. Service Dependencies

### 8.1 External Services

| Service | Purpose | Status | Fallback |
|---------|---------|--------|----------|
| Anthropic API | LLM provider | Required | None |
| Tavily | Web search | Optional | Disabled |
| ElevenLabs | Voice synthesis | Optional | Disabled |
| Twilio | SMS/Voice | Optional | Disabled |
| Redis | Caching | Optional | In-memory |
| PostgreSQL | State storage | Optional | File-based |
| Qdrant | Vector search | Optional | File-based |

### 8.2 Health Checks

```python
# sigil/health.py

async def check_health() -> dict:
    """Check health of all dependencies."""
    health = {
        "status": "healthy",
        "checks": {}
    }

    # Check LLM provider
    try:
        # Quick ping to Anthropic
        health["checks"]["llm"] = {"status": "healthy"}
    except Exception as e:
        health["checks"]["llm"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # Check database
    try:
        # Quick query
        health["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "unhealthy"

    # Check Redis
    try:
        # Ping Redis
        health["checks"]["cache"] = {"status": "healthy"}
    except Exception as e:
        health["checks"]["cache"] = {"status": "degraded", "error": str(e)}

    return health
```

### 8.3 Circuit Breaker Pattern

```python
# sigil/resilience.py

from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_llm(prompt: str) -> str:
    """Call LLM with circuit breaker protection."""
    return await llm_client.complete(prompt)
```

---

## 9. Monitoring & Observability

### 9.1 Logging Configuration

```python
# sigil/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter for production."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id

        if hasattr(record, "agent_id"):
            log_entry["agent_id"] = record.agent_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str = "INFO", format: str = "json"):
    """Configure logging for the application."""
    formatter = JSONFormatter() if format == "json" else logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.root.setLevel(getattr(logging, level.upper()))
    logging.root.handlers = [handler]
```

### 9.2 Metrics Collection

```python
# sigil/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    "sigil_requests_total",
    "Total request count",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "sigil_request_latency_seconds",
    "Request latency",
    ["endpoint", "method"]
)

# LLM metrics
LLM_TOKENS_USED = Counter(
    "sigil_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "model", "type"]
)

LLM_LATENCY = Histogram(
    "sigil_llm_latency_seconds",
    "LLM call latency",
    ["provider", "model"]
)

# Memory metrics
MEMORY_ITEMS = Gauge(
    "sigil_memory_items",
    "Number of memory items",
    ["category"]
)

# Contract metrics
CONTRACT_VALIDATIONS = Counter(
    "sigil_contract_validations_total",
    "Contract validation count",
    ["contract_name", "result"]
)
```

### 9.3 Distributed Tracing

```python
# sigil/tracing.py

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing(service_name: str, endpoint: str):
    """Configure OpenTelemetry tracing."""
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)


# Usage example
tracer = setup_tracing("sigil-api", "http://jaeger:4317")

async def execute_with_tracing(task: str):
    with tracer.start_as_current_span("execute_task") as span:
        span.set_attribute("task", task)
        result = await execute(task)
        span.set_attribute("result_valid", result.is_valid)
        return result
```

### 9.4 Alerting Rules

```yaml
# prometheus/alerts.yml

groups:
  - name: sigil-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(sigil_requests_total{status="5xx"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: Error rate is {{ $value | printf "%.2f" }} errors/sec

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(sigil_request_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High latency detected
          description: 95th percentile latency is {{ $value | printf "%.2f" }}s

      - alert: TokenBudgetNearLimit
        expr: sigil_llm_tokens_total > 200000
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: Token budget approaching limit
          description: Total tokens used: {{ $value }}
```

---

## 10. Security Configuration

### 10.1 JWT Authentication Setup

```python
# sigil/auth/jwt.py

import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def create_token(user_id: str, scopes: list[str]) -> str:
    """Create JWT token."""
    payload = {
        "sub": user_id,
        "scopes": scopes,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=60),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
```

### 10.2 Rate Limiting

```python
# sigil/middleware/rate_limit.py

from fastapi import Request, HTTPException
import time
from collections import defaultdict

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: int = 60, per: int = 60):
        self.rate = rate  # requests per period
        self.per = per    # period in seconds
        self.buckets = defaultdict(lambda: {"tokens": rate, "last": time.time()})

    def is_allowed(self, key: str) -> bool:
        bucket = self.buckets[key]
        now = time.time()

        # Refill tokens
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(
            self.rate,
            bucket["tokens"] + (elapsed * self.rate / self.per)
        )
        bucket["last"] = now

        # Check and consume
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


limiter = RateLimiter(rate=100, per=60)

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host
    if not limiter.is_allowed(client_ip):
        raise HTTPException(429, "Rate limit exceeded")
    return await call_next(request)
```

### 10.3 Input Sanitization

```python
# sigil/security/sanitize.py

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"system\s+prompt:",
    r"<\s*system\s*>",
    r"</\s*system\s*>",
    r"role:\s*(system|assistant)",
]

def sanitize_user_input(text: str) -> str:
    """Remove potential prompt injection patterns."""
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def validate_contract_name(name: str) -> bool:
    """Validate contract name is safe."""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$", name):
        return False
    return True
```

### 10.4 Secrets Management

```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
    --name sigil/production/api-keys \
    --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'

# Using HashiCorp Vault
vault kv put secret/sigil/production \
    ANTHROPIC_API_KEY=sk-ant-... \
    JWT_SECRET=...
```

---

## 11. Scaling Guide

### 11.1 Horizontal Scaling Strategy

```
                     Load Balancer
                          |
         +----------------+----------------+
         |                |                |
    +--------+       +--------+       +--------+
    | API 1  |       | API 2  |       | API N  |
    +--------+       +--------+       +--------+
         |                |                |
         +----------------+----------------+
                          |
    +---------------------------------------------+
    |              Shared Services                 |
    |                                              |
    |  +--------+  +----------+  +-----------+    |
    |  | Redis  |  | Postgres |  |  Qdrant   |    |
    |  +--------+  +----------+  +-----------+    |
    +---------------------------------------------+
```

### 11.2 Scaling Thresholds

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU Utilization | > 70% | < 30% |
| Memory Utilization | > 80% | < 40% |
| Request Latency (p95) | > 2s | < 500ms |
| Queue Depth | > 100 | < 10 |

### 11.3 Connection Pooling

```python
# sigil/db/pool.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)

async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue: Token Budget Exceeded**
```
Error: TokenBudgetExceeded - Total tokens 258000 exceeds budget 256000
```
**Solution:**
1. Check contract max_total_tokens setting
2. Review task complexity
3. Consider using simpler reasoning strategy
4. Enable context compression

**Issue: Contract Validation Failures**
```
Error: ContractValidationError - Missing required field: score
```
**Solution:**
1. Review agent system prompt for output format
2. Check deliverable definitions
3. Verify LLM model capabilities
4. Review retry/fallback settings

**Issue: Memory Retrieval Slow**
```
Warning: Memory retrieval took 5000ms (threshold: 2000ms)
```
**Solution:**
1. Switch to RAG mode instead of LLM
2. Add vector database
3. Reduce retrieval k parameter
4. Add caching layer

### 12.2 Debug Mode

```bash
# Enable debug logging
export SIGIL_LOG_LEVEL=DEBUG
export SIGIL_API_DEBUG=true

# Start with debug endpoints
python -m sigil.interfaces.api --debug

# Access debug endpoints
curl http://localhost:8000/debug/config
curl http://localhost:8000/debug/state
```

### 12.3 Log Analysis

```bash
# Search for errors
grep -i error logs/sigil.log | jq .

# Find slow requests
jq 'select(.latency_ms > 2000)' logs/sigil.log

# Count by error type
jq -r '.error_type' logs/sigil.log | sort | uniq -c | sort -rn

# Session timeline
jq -r 'select(.session_id == "sess-123")' logs/sigil.log | jq -s 'sort_by(.timestamp)'
```

### 12.4 Performance Profiling

```python
# sigil/profiling.py

import cProfile
import pstats
from functools import wraps

def profile(func):
    """Profile a function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = await func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)
        return result
    return wrapper
```

---

## 13. Maintenance Operations

### 13.1 Backup Procedures

```bash
# Backup PostgreSQL
pg_dump -h postgres -U sigil sigil > backup_$(date +%Y%m%d).sql

# Backup event store (file-based)
tar -czf events_backup_$(date +%Y%m%d).tar.gz data/state/

# Backup memory data
tar -czf memory_backup_$(date +%Y%m%d).tar.gz data/memory/

# Backup to S3
aws s3 cp backup_*.sql s3://sigil-backups/daily/
aws s3 cp events_backup_*.tar.gz s3://sigil-backups/daily/
aws s3 cp memory_backup_*.tar.gz s3://sigil-backups/daily/
```

### 13.2 Restore Procedures

```bash
# Restore PostgreSQL
psql -h postgres -U sigil sigil < backup_20260111.sql

# Restore event store
tar -xzf events_backup_20260111.tar.gz -C data/state/

# Restore memory data
tar -xzf memory_backup_20260111.tar.gz -C data/memory/
```

### 13.3 Log Rotation

```bash
# /etc/logrotate.d/sigil
/app/logs/sigil.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 sigil sigil
    postrotate
        kill -USR1 $(cat /app/sigil.pid) 2>/dev/null || true
    endscript
}
```

### 13.4 Health Monitoring Script

```bash
#!/bin/bash
# scripts/health-check.sh

ENDPOINT="http://localhost:8000/health"
SLACK_WEBHOOK="https://hooks.slack.com/services/..."

response=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ "$response" != "200" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"ALERT: Sigil health check failed"}' \
        $SLACK_WEBHOOK
fi
```

---

## 14. Disaster Recovery

### 14.1 Recovery Time Objectives

| Scenario | RTO | RPO |
|----------|-----|-----|
| Single service failure | 5 min | 0 |
| Database failure | 30 min | 5 min |
| Complete system failure | 2 hours | 1 hour |
| Data center failure | 4 hours | 1 hour |

### 14.2 Recovery Procedures

**Single Service Failure:**
```bash
# Kubernetes auto-recovery
kubectl rollout restart deployment/sigil-api

# Docker Compose recovery
docker-compose restart sigil-api
```

**Database Failure:**
```bash
# Switch to replica
ALTER SYSTEM SET primary_conninfo = 'host=postgres-replica';
SELECT pg_promote();

# Restore from backup if needed
pg_restore -h postgres -U sigil -d sigil backup.dump
```

**Complete System Recovery:**
```bash
# 1. Restore database
pg_restore -h postgres -U sigil -d sigil latest_backup.dump

# 2. Restore event store
tar -xzf events_backup.tar.gz -C /app/data/state/

# 3. Restart services
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health

# 5. Run integrity check
python -m sigil.maintenance.verify_integrity
```

### 14.3 Failover Configuration

```yaml
# PostgreSQL replication
primary:
  host: postgres-primary
  port: 5432

replica:
  host: postgres-replica
  port: 5432

failover:
  automatic: true
  health_check_interval: 10s
  failover_threshold: 3
```

---

## 15. Appendices

### 15.1 Appendix A: Full Docker Compose (Production)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  sigil-api:
    image: your-registry/sigil:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - SIGIL_ENVIRONMENT=production
    secrets:
      - anthropic_api_key
      - jwt_secret
    networks:
      - sigil-internal
      - sigil-external

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - sigil-external

secrets:
  anthropic_api_key:
    external: true
  jwt_secret:
    external: true

networks:
  sigil-internal:
    internal: true
  sigil-external:
```

### 15.2 Appendix B: Kubernetes ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sigil-config
  namespace: sigil
data:
  SIGIL_USE_MEMORY: "true"
  SIGIL_USE_PLANNING: "true"
  SIGIL_USE_CONTRACTS: "true"
  SIGIL_USE_ROUTING: "true"
  SIGIL_API_HOST: "0.0.0.0"
  SIGIL_API_PORT: "8000"
  SIGIL_LOG_LEVEL: "INFO"
  SIGIL_LOG_FORMAT: "json"
```

### 15.3 Appendix C: Terraform Infrastructure

```hcl
# terraform/main.tf

provider "aws" {
  region = "us-east-1"
}

module "sigil_vpc" {
  source = "./modules/vpc"
  name   = "sigil-vpc"
}

module "sigil_ecs" {
  source     = "./modules/ecs"
  name       = "sigil-cluster"
  vpc_id     = module.sigil_vpc.vpc_id
  subnets    = module.sigil_vpc.private_subnets
}

module "sigil_rds" {
  source        = "./modules/rds"
  name          = "sigil-db"
  engine        = "postgres"
  engine_version = "15"
  instance_class = "db.r6g.large"
  vpc_id        = module.sigil_vpc.vpc_id
  subnets       = module.sigil_vpc.private_subnets
}
```

### 15.4 Appendix D: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0-alpha | 2026-01-11 | Initial Phase 7 release |
| 1.0.0 | 2025-12-01 | v1 foundation release |

---

**Document End**

*Generated: 2026-01-11*
*Deployment Guide: Sigil v2 Phase 7*
