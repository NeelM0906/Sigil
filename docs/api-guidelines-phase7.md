# Phase 7 API Guidelines: Integration & Polish

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Last Updated | 2026-01-11 |
| Authors | API Architecture Team |

---

## Table of Contents

1. [Overview](#overview)
2. [Naming Conventions](#naming-conventions)
3. [Request Guidelines](#request-guidelines)
4. [Response Guidelines](#response-guidelines)
5. [Error Handling](#error-handling)
6. [Authentication & Authorization](#authentication--authorization)
7. [Rate Limiting](#rate-limiting)
8. [Caching](#caching)
9. [WebSocket Guidelines](#websocket-guidelines)
10. [Security Best Practices](#security-best-practices)
11. [Testing Guidelines](#testing-guidelines)
12. [Integration Patterns](#integration-patterns)
13. [Performance Guidelines](#performance-guidelines)
14. [Versioning](#versioning)

---

## Overview

This document provides comprehensive guidelines for implementing and consuming the Phase 7 Integration API. These guidelines ensure consistency, reliability, and maintainability across all API interactions.

### Core Principles

1. **Consistency**: All endpoints follow the same patterns and conventions
2. **Predictability**: Behavior is documented and consistent across operations
3. **Discoverability**: APIs are self-documenting with clear naming
4. **Robustness**: Error handling is comprehensive and actionable
5. **Performance**: Optimizations are built-in, not afterthoughts

### Target Audience

- Backend developers implementing the API
- Frontend developers consuming the API
- DevOps engineers deploying and monitoring
- QA engineers testing the API

---

## Naming Conventions

### URL Patterns

```
Pattern: /v{version}/{resource}[/{identifier}][/{sub-resource}]

Examples:
  GET    /v7/agents                    # List agents
  POST   /v7/agents                    # Create agent
  GET    /v7/agents/{name}             # Get specific agent
  PATCH  /v7/agents/{name}             # Update agent
  DELETE /v7/agents/{name}             # Delete agent
  POST   /v7/agents/{name}/run         # Action on agent
  GET    /v7/agents/{name}/status      # Sub-resource
```

### Resource Naming

| Pattern | Convention | Example |
|---------|------------|---------|
| Collections | Plural nouns | `/agents`, `/sessions` |
| Individual | Singular via identifier | `/agents/{name}` |
| Actions | Verb as sub-resource | `/agents/{name}/run` |
| Sub-resources | Nested path | `/agents/{name}/versions` |

### Field Naming

```python
# Python/JSON: Use snake_case for all fields
{
    "agent_name": "sales_qualifier",
    "created_at": "2026-01-11T12:00:00Z",
    "system_prompt": "...",
    "max_tokens": 4000,
    "is_active": true
}

# TypeScript interfaces: Match JSON schema
interface AgentConfig {
    agent_name: string;
    created_at: string;
    system_prompt: string;
    max_tokens: number;
    is_active: boolean;
}
```

### Class Naming

```python
# Python classes: Use PascalCase
class SigilOrchestrator:
    pass

class ContextManager:
    pass

class TokenBudgetExceededError(SigilError):
    pass

# Enum values: Use SCREAMING_SNAKE_CASE
class OperationType(str, Enum):
    CREATE = "create"
    RUN = "run"
    EVALUATE = "evaluate"
```

### Error Codes

```
Format: {COMPONENT}_{CATEGORY}_{NUMBER}

Components:
  ORCH - Orchestrator
  CTX  - Context Manager
  EVO  - Evolution Manager
  AGT  - Agent
  MEM  - Memory
  WS   - WebSocket
  RATE - Rate Limiting

Examples:
  ORCH_001 - Orchestrator not initialized
  AGT_002  - Agent creation failed
  CTX_003  - Token budget exceeded
```

---

## Request Guidelines

### Required Headers

```http
# Always required
Authorization: Bearer <jwt_token>
Content-Type: application/json
Accept: application/json

# Recommended for tracing
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Session-ID: session_abc123

# Optional client identification
X-Client-ID: web-app-v2
X-Client-Version: 2.1.0
```

### Request ID Generation

```python
import uuid
from datetime import datetime

def generate_request_id() -> str:
    """
    Generate a unique request ID.

    Format: UUID v4
    """
    return str(uuid.uuid4())

# Example usage
request_id = generate_request_id()
headers = {
    "X-Request-ID": request_id,
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
```

### Request Body Format

```python
# Standard request structure
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",  # Optional, auto-generated if omitted
    "operation": "run",
    "payload": {
        # Operation-specific fields
    },
    "config": {
        # Optional configuration overrides
    },
    "metadata": {
        # Optional client metadata
    }
}
```

### Pagination

```http
# Request
GET /v7/agents?page=2&per_page=20

# Response includes pagination info
{
    "agents": [...],
    "pagination": {
        "page": 2,
        "per_page": 20,
        "total": 57,
        "total_pages": 3
    }
}
```

### Filtering

```http
# Single filter
GET /v7/agents?stratum=RAI

# Multiple filters
GET /v7/agents?stratum=RAI&status=active

# Date range
GET /v7/agents?created_after=2026-01-01&created_before=2026-01-31

# Search
GET /v7/agents?q=sales
```

### Sorting

```http
# Ascending (default)
GET /v7/agents?sort=created_at

# Descending (prefix with -)
GET /v7/agents?sort=-created_at

# Multiple sort fields
GET /v7/agents?sort=-created_at,name
```

---

## Response Guidelines

### Success Response Format

```python
# 200 OK - Resource returned
{
    "agent_name": "sales_qualifier",
    "version": "1.2.0",
    "status": "active",
    "metadata": {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "tokens_used": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        },
        "latency_ms": 250
    }
}

# 201 Created - Resource created
{
    "agent_name": "new_agent",
    "version": "1.0.0",
    "created_at": "2026-01-11T12:00:00Z",
    "metadata": {...}
}

# 204 No Content - Success with no body
# (No response body)
```

### List Response Format

```python
{
    "agents": [
        {"name": "agent1", ...},
        {"name": "agent2", ...}
    ],
    "pagination": {
        "page": 1,
        "per_page": 20,
        "total": 57,
        "total_pages": 3
    },
    "metadata": {
        "request_id": "...",
        "latency_ms": 45
    }
}
```

### Response Headers

```http
# Always included
Content-Type: application/json
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000

# Rate limiting
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704978000

# Caching (when applicable)
Cache-Control: private, max-age=60
ETag: "abc123"

# Token tracking
X-Tokens-Used: 150
X-Tokens-Remaining: 99850
```

### HTTP Status Codes

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PATCH, POST (non-create) |
| 201 | Created | Successful POST creating a resource |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid request format or validation error |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Valid auth but insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 408 | Request Timeout | Request took too long |
| 409 | Conflict | Resource conflict (e.g., duplicate name) |
| 422 | Unprocessable | Semantic validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Error Handling

### Error Response Format (RFC 9457)

```python
# Standard error response
{
    "type": "https://sigil.acti.ai/errors/agt-001",
    "title": "AgentNotFoundError",
    "status": 404,
    "detail": "Agent 'unknown_agent' not found",
    "instance": "/v7/agents/unknown_agent",
    "error_code": "AGT_001",
    "timestamp": "2026-01-11T12:00:00Z",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "recovery_hints": [
        "Verify agent name is correct",
        "Create the agent if it doesn't exist",
        "List available agents with GET /agents"
    ]
}

# Validation error with details
{
    "type": "https://sigil.acti.ai/errors/req-001",
    "title": "RequestValidationError",
    "status": 400,
    "detail": "Request validation failed",
    "instance": "/v7/agents",
    "error_code": "REQ_001",
    "timestamp": "2026-01-11T12:00:00Z",
    "request_id": "550e8400-e29b-41d4-a716-446655440001",
    "validation_errors": [
        {"field": "name", "message": "Agent name must be 3-64 characters"},
        {"field": "stratum", "message": "Invalid stratum, must be one of: RTI, RAI, ZACS, EEI, IGE"}
    ],
    "recovery_hints": [
        "Check field validation requirements",
        "Refer to API documentation for valid values"
    ]
}
```

### Error Categories

```python
# Client Errors (4xx) - Client should fix and retry
400 Bad Request       # Malformed request
401 Unauthorized      # Auth required
403 Forbidden         # Insufficient permissions
404 Not Found         # Resource doesn't exist
408 Request Timeout   # Client can retry
409 Conflict          # Resolve conflict first
422 Unprocessable     # Fix validation errors
429 Rate Limited      # Wait and retry

# Server Errors (5xx) - Client can retry with backoff
500 Internal Error    # Unexpected error
502 Bad Gateway       # Upstream service error
503 Unavailable       # Temporary unavailability
504 Gateway Timeout   # Upstream timeout
```

### Client Error Handling

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class SigilClient:
    """Example client with proper error handling."""

    def __init__(self, base_url: str, token: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"}
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=lambda e: isinstance(e, RetryableError)
    )
    async def run_agent(self, agent_name: str, input: str) -> dict:
        """Run agent with retry logic."""
        try:
            response = await self.client.post(
                f"/v7/agents/{agent_name}/run",
                json={"input": input}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error = e.response.json()

            if e.response.status_code == 429:
                # Rate limited - extract retry info
                retry_after = int(e.response.headers.get("Retry-After", 30))
                raise RateLimitError(retry_after=retry_after)

            elif e.response.status_code == 408:
                # Timeout - retryable
                raise RetryableError(error["detail"])

            elif e.response.status_code >= 500:
                # Server error - retryable
                raise RetryableError(error["detail"])

            else:
                # Client error - not retryable
                raise ClientError(
                    error_code=error["error_code"],
                    message=error["detail"],
                    recovery_hints=error.get("recovery_hints", [])
                )
```

---

## Authentication & Authorization

### Authentication Methods

```python
# JWT Bearer Token (Primary)
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."
}

# API Key (Service-to-service)
headers = {
    "X-API-Key": "sk_live_abc123..."
}
```

### Token Format

```python
# JWT payload structure
{
    "sub": "user_123",              # Subject (user ID)
    "aud": "sigil-api",             # Audience
    "iss": "https://auth.sigil.acti.ai",  # Issuer
    "exp": 1704978000,              # Expiration
    "iat": 1704974400,              # Issued at
    "scope": "agents:read agents:write evolution:read"
}
```

### Permission Scopes

| Scope | Description |
|-------|-------------|
| `agents:read` | List and view agents |
| `agents:write` | Create, update, delete agents |
| `agents:execute` | Run agents |
| `memory:read` | Search memory |
| `memory:write` | Store memory |
| `evolution:read` | View evaluations |
| `evolution:write` | Optimize agents |
| `admin` | Full access |

### Authorization Checks

```python
from functools import wraps
from fastapi import Depends, HTTPException, status

def require_scope(*required_scopes: str):
    """Decorator to require specific scopes."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            user_scopes = set(user.scopes)
            required = set(required_scopes)

            if not required.issubset(user_scopes):
                missing = required - user_scopes
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scopes: {', '.join(missing)}"
                )

            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator

# Usage
@app.post("/v7/agents")
@require_scope("agents:write")
async def create_agent(request: CreateAgentRequest, user: User):
    ...
```

---

## Rate Limiting

### Rate Limit Tiers

| Tier | Requests/sec | Requests/min | Requests/hour | Tokens/min |
|------|--------------|--------------|---------------|------------|
| Free | 1 | 20 | 100 | 10,000 |
| Standard | 10 | 100 | 1,000 | 100,000 |
| Pro | 50 | 500 | 5,000 | 500,000 |
| Enterprise | 200 | 2,000 | 20,000 | 2,000,000 |

### Rate Limit Headers

```http
# Response headers
X-RateLimit-Limit: 100          # Max requests in window
X-RateLimit-Remaining: 95       # Remaining requests
X-RateLimit-Reset: 1704978000   # Unix timestamp when limit resets
X-RateLimit-Window: 60          # Window size in seconds

# When rate limited (429)
Retry-After: 30                 # Seconds until retry allowed
```

### Handling Rate Limits

```python
import asyncio
from datetime import datetime

class RateLimitHandler:
    """Handle rate limiting gracefully."""

    async def make_request(self, client, method, url, **kwargs):
        """Make request with rate limit handling."""
        while True:
            response = await client.request(method, url, **kwargs)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 30))
                logger.warning(
                    "Rate limited, waiting",
                    retry_after=retry_after
                )
                await asyncio.sleep(retry_after)
                continue

            # Track remaining quota
            remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            if remaining < 10:
                logger.warning(
                    "Rate limit quota low",
                    remaining=remaining
                )

            return response
```

### Rate Limit Best Practices

1. **Monitor headers**: Track `X-RateLimit-Remaining` proactively
2. **Implement backoff**: Use exponential backoff on 429 errors
3. **Batch requests**: Combine multiple operations when possible
4. **Use webhooks**: For async operations, use webhooks instead of polling
5. **Cache responses**: Reduce redundant requests with caching

---

## Caching

### Cacheable Endpoints

| Endpoint | Cache-Control | TTL | Notes |
|----------|---------------|-----|-------|
| `GET /agents` | `private, max-age=60` | 60s | Invalidate on create/update |
| `GET /agents/{name}` | `private, max-age=300` | 5min | Invalidate on update |
| `GET /agents/{name}/status` | `private, max-age=10` | 10s | Frequent updates |
| `POST /memory/search` | `private, max-age=30` | 30s | Same query = cached |
| `GET /health` | `no-cache` | - | Always fresh |

### Cache Headers

```http
# Cacheable response
Cache-Control: private, max-age=60
ETag: "abc123def456"
Last-Modified: Sat, 11 Jan 2026 12:00:00 GMT

# Conditional request
If-None-Match: "abc123def456"
If-Modified-Since: Sat, 11 Jan 2026 12:00:00 GMT

# 304 Not Modified response (cache hit)
HTTP/1.1 304 Not Modified
ETag: "abc123def456"
```

### Client-Side Caching

```python
import hashlib
from datetime import datetime, timedelta

class CachingClient:
    """Client with built-in caching."""

    def __init__(self):
        self._cache = {}

    def _cache_key(self, method: str, url: str, params: dict) -> str:
        """Generate cache key."""
        key_data = f"{method}:{url}:{sorted(params.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, url: str, params: dict = None) -> dict:
        """GET with caching."""
        cache_key = self._cache_key("GET", url, params or {})

        # Check cache
        cached = self._cache.get(cache_key)
        if cached and cached["expires_at"] > datetime.utcnow():
            return cached["data"]

        # Make request with conditional headers
        headers = {}
        if cached:
            headers["If-None-Match"] = cached.get("etag")

        response = await self.client.get(url, params=params, headers=headers)

        if response.status_code == 304:
            # Cache still valid
            return cached["data"]

        # Parse cache headers
        data = response.json()
        cache_control = response.headers.get("Cache-Control", "")
        max_age = self._parse_max_age(cache_control)
        etag = response.headers.get("ETag")

        if max_age > 0:
            self._cache[cache_key] = {
                "data": data,
                "etag": etag,
                "expires_at": datetime.utcnow() + timedelta(seconds=max_age)
            }

        return data
```

---

## WebSocket Guidelines

### Connection Lifecycle

```javascript
// Client connection example
const ws = new WebSocket('wss://api.sigil.acti.ai/v7/ws/agents/sales_qualifier/run');

ws.onopen = () => {
    console.log('Connected');

    // Send run request
    ws.send(JSON.stringify({
        type: 'run_request',
        input: 'Qualify this lead: John from Acme',
        session_id: 'session_123',
        config: {
            timeout_ms: 30000
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch (data.type) {
        case 'stream_start':
            console.log('Processing started');
            break;
        case 'token':
            process.stdout.write(data.data.content);
            break;
        case 'tool_call':
            console.log(`Tool: ${data.data.tool_name}`);
            break;
        case 'complete':
            console.log('\\nCompleted');
            console.log(`Tokens: ${data.data.tokens_used.total_tokens}`);
            break;
        case 'error':
            console.error(`Error: ${data.data.error_message}`);
            break;
        case 'backpressure':
            // Pause sending, wait for queue to drain
            console.log('Backpressure - pausing');
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    console.log(`Closed: ${event.code} - ${event.reason}`);
};
```

### Message Types

```python
# Client -> Server messages
{
    "type": "run_request",
    "input": "...",
    "session_id": "...",
    "config": {...}
}

{
    "type": "ready",          # Acknowledge backpressure, ready for more
    "last_sequence": 42       # Last received sequence number
}

{
    "type": "cancel"          # Cancel current operation
}

{
    "type": "ping"            # Keepalive
}

# Server -> Client messages
{
    "type": "connection_ready",
    "data": {"status": "ready", "message": "Connected to agent: sales_qualifier"},
    "timestamp": "2026-01-11T12:00:00Z",
    "sequence": 0
}

{
    "type": "token",
    "data": {"content": "Hello", "index": 0},
    "timestamp": "2026-01-11T12:00:01Z",
    "sequence": 1
}

{
    "type": "tool_call",
    "data": {"tool_id": "tc_001", "tool_name": "crm", "arguments": {...}},
    "timestamp": "2026-01-11T12:00:02Z",
    "sequence": 10
}

{
    "type": "backpressure",
    "data": {"queue_size": 85, "max_queue_size": 100, "paused": true},
    "timestamp": "2026-01-11T12:00:03Z",
    "sequence": 50
}

{
    "type": "complete",
    "data": {"request_id": "...", "tokens_used": {...}, "latency_ms": 3500},
    "timestamp": "2026-01-11T12:00:05Z",
    "sequence": 75
}
```

### Backpressure Handling

```python
class WebSocketClient:
    """WebSocket client with backpressure handling."""

    def __init__(self, url: str):
        self.url = url
        self.paused = False
        self.pending_ready = False

    async def handle_message(self, message: dict):
        """Handle incoming messages."""
        msg_type = message["type"]

        if msg_type == "backpressure":
            # Server is pausing - stop requesting more
            self.paused = True
            logger.warning(
                "Backpressure received",
                queue_size=message["data"]["queue_size"]
            )

        elif msg_type == "token":
            # Process token
            await self.process_token(message["data"])

            # If we were paused and processed successfully, signal ready
            if self.paused and not self.pending_ready:
                self.pending_ready = True
                await self.send({
                    "type": "ready",
                    "last_sequence": message["sequence"]
                })
                self.paused = False
                self.pending_ready = False

    async def process_token(self, data: dict):
        """Process received token."""
        # Simulate slow processing
        await asyncio.sleep(0.01)
        print(data["content"], end="", flush=True)
```

### Reconnection Strategy

```python
class ReconnectingWebSocket:
    """WebSocket with automatic reconnection."""

    def __init__(self, url: str, max_retries: int = 5):
        self.url = url
        self.max_retries = max_retries
        self.retry_count = 0
        self.base_delay = 1.0
        self.max_delay = 30.0

    async def connect(self):
        """Connect with exponential backoff."""
        while self.retry_count < self.max_retries:
            try:
                self.ws = await websockets.connect(self.url)
                self.retry_count = 0  # Reset on success
                return
            except Exception as e:
                self.retry_count += 1
                delay = min(
                    self.base_delay * (2 ** self.retry_count),
                    self.max_delay
                )
                logger.warning(
                    "WebSocket connection failed",
                    retry=self.retry_count,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)

        raise MaxRetriesExceeded("Failed to connect after max retries")
```

---

## Security Best Practices

### Input Validation

```python
from pydantic import BaseModel, Field, validator
import re

class CreateAgentRequest(BaseModel):
    """Request model with validation."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r'^[a-z][a-z0-9_]*$'
    )
    stratum: str = Field(..., pattern=r'^(RTI|RAI|ZACS|EEI|IGE)$')
    description: str = Field(..., max_length=500)
    system_prompt: str = Field(None, max_length=50000)

    @validator('name')
    def validate_name(cls, v):
        """Validate agent name."""
        # Check for reserved names
        reserved = {'admin', 'system', 'root', 'api'}
        if v.lower() in reserved:
            raise ValueError(f"Name '{v}' is reserved")

        # Check for SQL injection patterns
        if re.search(r'[;\'"\\]', v):
            raise ValueError("Invalid characters in name")

        return v

    @validator('system_prompt')
    def validate_prompt(cls, v):
        """Validate system prompt."""
        if v is None:
            return v

        # Check for prompt injection attempts
        dangerous_patterns = [
            r'ignore\s+(previous|above)\s+instructions',
            r'system:\s*',
            r'<\|im_start\|>',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid content in system prompt")

        return v
```

### Output Sanitization

```python
import html
from typing import Any

def sanitize_output(data: Any) -> Any:
    """Sanitize output data."""
    if isinstance(data, str):
        # Escape HTML entities
        return html.escape(data)
    elif isinstance(data, dict):
        return {k: sanitize_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    else:
        return data

def redact_sensitive(data: dict, fields: set) -> dict:
    """Redact sensitive fields."""
    result = {}
    for key, value in data.items():
        if key in fields:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_sensitive(value, fields)
        else:
            result[key] = value
    return result

# Sensitive fields to redact in logs
SENSITIVE_FIELDS = {
    'api_key', 'token', 'password', 'secret',
    'authorization', 'credentials', 'private_key'
}
```

### Secret Management

```python
# Never log secrets
import structlog

class SecretFilter:
    """Filter secrets from logs."""

    def __call__(self, logger, method_name, event_dict):
        for key in list(event_dict.keys()):
            if any(s in key.lower() for s in ['key', 'token', 'secret', 'password']):
                event_dict[key] = "[REDACTED]"
        return event_dict

# Configure structured logging
structlog.configure(
    processors=[
        SecretFilter(),
        structlog.processors.JSONRenderer()
    ]
)
```

### CORS Configuration

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.sigil.acti.ai",
        "https://console.sigil.acti.ai",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600,  # Cache preflight for 1 hour
)
```

---

## Testing Guidelines

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestSigilOrchestrator:
    """Unit tests for SigilOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        config = OrchestratorConfig(
            max_concurrent_requests=10,
            max_input_tokens=150000,
        )
        return SigilOrchestrator(config)

    @pytest.fixture
    async def initialized_orchestrator(self, orchestrator):
        """Create initialized orchestrator."""
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_handle_validates_request(self, initialized_orchestrator):
        """Test that handle validates requests."""
        request = OrchestratorRequest(
            request_id="invalid",  # Not a valid UUID
            operation=OperationType.RUN,
            payload=RunAgentPayload(
                agent_name="test",
                input="Hello",
            ),
        )

        response = await initialized_orchestrator.handle(request)

        assert not response.success
        assert response.error.error_code == "ORCH_VALIDATION_001"

    @pytest.mark.asyncio
    async def test_handle_respects_timeout(self, initialized_orchestrator):
        """Test that handle respects timeout configuration."""
        request = OrchestratorRequest(
            request_id=str(uuid.uuid4()),
            operation=OperationType.RUN,
            payload=RunAgentPayload(
                agent_name="slow_agent",
                input="Hello",
            ),
            config=RequestConfig(timeout_ms=100),  # Very short timeout
        )

        with patch.object(
            initialized_orchestrator,
            '_handle_run',
            new=AsyncMock(side_effect=asyncio.sleep(1))  # Slow handler
        ):
            response = await initialized_orchestrator.handle(request)

        assert not response.success
        assert response.error.error_code == "ORCH_TIMEOUT_001"
```

### Integration Testing

```python
import pytest
from httpx import AsyncClient

class TestAgentEndpoints:
    """Integration tests for agent endpoints."""

    @pytest.fixture
    async def client(self):
        """Create test client."""
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_create_agent(self, client, auth_headers):
        """Test agent creation."""
        response = await client.post(
            "/v7/agents",
            json={
                "name": "test_agent",
                "stratum": "RAI",
                "description": "Test agent",
            },
            headers=auth_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_name"] == "test_agent"
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_run_agent(self, client, auth_headers, created_agent):
        """Test running an agent."""
        response = await client.post(
            f"/v7/agents/{created_agent}/run",
            json={"input": "Test input"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "metadata" in data
        assert data["metadata"]["tokens_used"]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting behavior."""
        # Make many requests quickly
        responses = await asyncio.gather(*[
            client.get("/v7/agents", headers=auth_headers)
            for _ in range(150)  # Exceed limit
        ])

        # Some should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes

        # Check headers on rate limited response
        limited = next(r for r in responses if r.status_code == 429)
        assert "Retry-After" in limited.headers
        assert "X-RateLimit-Remaining" in limited.headers
```

### Contract Testing

```python
from schemathesis import from_openapi

schema = from_openapi("/docs/openapi-phase7-integration.yaml")

@schema.parametrize()
def test_api_contract(case):
    """Test all endpoints match OpenAPI contract."""
    response = case.call()
    case.validate_response(response)
```

---

## Integration Patterns

### Sync vs Async Operations

```python
# Synchronous - Use for quick operations
response = await client.post("/v7/agents", json={...})
# Response includes result immediately

# Asynchronous - Use for long-running operations
response = await client.post("/v7/evolution/optimize", json={...})
# Response includes operation_id for polling
operation_id = response.json()["operation_id"]

# Poll for completion
while True:
    status = await client.get(f"/v7/operations/{operation_id}")
    if status.json()["status"] == "completed":
        result = status.json()["result"]
        break
    await asyncio.sleep(5)
```

### Webhook Integration

```python
# Register webhook
await client.post("/v7/webhooks", json={
    "url": "https://my-app.com/webhooks/sigil",
    "events": ["agent.run.completed", "evolution.optimization.completed"],
    "secret": "whsec_abc123...",
})

# Webhook handler (your server)
@app.post("/webhooks/sigil")
async def handle_webhook(request: Request):
    # Verify signature
    signature = request.headers.get("X-Sigil-Signature")
    payload = await request.body()

    if not verify_signature(payload, signature, WEBHOOK_SECRET):
        raise HTTPException(status_code=401)

    event = await request.json()

    if event["type"] == "agent.run.completed":
        await process_agent_result(event["data"])

    return {"received": True}
```

### Batch Operations

```python
# Batch request format
response = await client.post("/v7/batch", json={
    "requests": [
        {
            "id": "req_1",
            "method": "POST",
            "path": "/agents/agent1/run",
            "body": {"input": "Query 1"}
        },
        {
            "id": "req_2",
            "method": "POST",
            "path": "/agents/agent2/run",
            "body": {"input": "Query 2"}
        },
    ]
})

# Batch response
{
    "responses": [
        {"id": "req_1", "status": 200, "body": {...}},
        {"id": "req_2", "status": 200, "body": {...}},
    ]
}
```

---

## Performance Guidelines

### Request Optimization

```python
# DO: Specify only needed fields
await client.post("/v7/agents/sales/run", json={
    "input": "Quick query",
    "config": {
        "include_memory": False,      # Skip if not needed
        "include_history": False,     # Skip if not needed
        "memory_retrieval_k": 3,      # Reduce if possible
        "history_turns": 5,           # Reduce if possible
    }
})

# DON'T: Use defaults blindly
await client.post("/v7/agents/sales/run", json={
    "input": "Quick query"
    # Default retrieves 10 memory items, 10 history turns
})
```

### Token Budget Management

```python
# Monitor token usage
response = await client.post("/v7/agents/sales/run", json={"input": "..."})
tokens = response.json()["metadata"]["tokens_used"]

if tokens["total_tokens"] > 10000:
    logger.warning(
        "High token usage",
        total=tokens["total_tokens"],
        input=tokens["input_tokens"],
        output=tokens["output_tokens"]
    )

# Proactive budget checks
budget_status = await client.get("/v7/budget/status")
if budget_status.json()["utilization"] > 0.8:
    # Enable compression or reduce scope
    config["compression"] = True
```

### Connection Pooling

```python
# Configure connection pool
import httpx

# Good: Reuse client
client = httpx.AsyncClient(
    base_url="https://api.sigil.acti.ai/v7",
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0,
    ),
    timeout=httpx.Timeout(30.0, connect=5.0),
)

# Use throughout application lifecycle
async def main():
    async with client:
        # All requests share connection pool
        result1 = await client.get("/agents")
        result2 = await client.post("/agents/test/run", json={...})

# Bad: Creating new client per request
async def bad_example():
    async with httpx.AsyncClient() as client:  # New pool per request!
        return await client.get("https://api.sigil.acti.ai/v7/agents")
```

---

## Versioning

### API Version Strategy

```
URL-based versioning: /v{major}/...

Examples:
  /v7/agents     # Current version
  /v6/agents     # Previous version (deprecated)
  /v8/agents     # Next version (preview)
```

### Version Lifecycle

| Phase | Duration | Description |
|-------|----------|-------------|
| Preview | Variable | Beta features, may change |
| Current | 24 months | Stable, fully supported |
| Deprecated | 12 months | Supported but discouraged |
| Retired | - | No longer available |

### Deprecation Headers

```http
# Response for deprecated endpoint
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 11 Jan 2027 00:00:00 GMT
Link: </v8/agents>; rel="successor-version"
```

### Version Migration

```python
# Check for deprecation in client
response = await client.get("/v7/agents")

if response.headers.get("Deprecation"):
    sunset = response.headers.get("Sunset")
    successor = response.headers.get("Link")
    logger.warning(
        "Using deprecated API version",
        sunset=sunset,
        successor=successor
    )
```

---

## Appendix: Common Patterns

### Idempotency

```python
# Use idempotency key for POST requests
headers = {
    "Authorization": f"Bearer {token}",
    "X-Idempotency-Key": str(uuid.uuid4()),
}

# Same key = same result (even if called multiple times)
response1 = await client.post("/v7/agents", json=data, headers=headers)
response2 = await client.post("/v7/agents", json=data, headers=headers)
# response1 == response2
```

### Long-Running Operations

```python
# Start operation
response = await client.post("/v7/evolution/optimize", json={
    "agent_name": "sales_qualifier",
    "target": "accuracy"
})

operation_id = response.json()["operation_id"]

# Poll with exponential backoff
delay = 5
while True:
    status = await client.get(f"/v7/operations/{operation_id}")
    state = status.json()["state"]

    if state == "completed":
        return status.json()["result"]
    elif state == "failed":
        raise OperationFailed(status.json()["error"])
    elif state == "running":
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 60)  # Max 60s between polls
```

### Graceful Degradation

```python
async def run_agent_with_fallback(client, agent_name: str, input: str):
    """Run agent with graceful fallback."""
    try:
        response = await client.post(
            f"/v7/agents/{agent_name}/run",
            json={"input": input},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        # Fallback to simpler execution
        return await client.post(
            f"/v7/agents/{agent_name}/run",
            json={
                "input": input,
                "config": {
                    "include_memory": False,
                    "include_history": False,
                }
            },
            timeout=60.0
        ).json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # Service unavailable - use cached response
            return get_cached_response(agent_name, input)
        raise
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | API Architecture Team | Initial Phase 7 guidelines |

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-11*
