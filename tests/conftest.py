"""
Pytest configuration and shared fixtures for VaaS tests.
"""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    """Calculate n factorial."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test the functions
if __name__ == "__main__":
    print(f"fib(10) = {fibonacci(10)}")
    print(f"fact(5) = {factorial(5)}")
'''


@pytest.fixture
def sample_terraform_code():
    """Sample Terraform code for testing."""
    return '''
resource "aws_s3_bucket" "main" {
  bucket = "my-application-bucket"
  acl    = "private"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Environment = "production"
    Owner       = "platform-team"
    Project     = "main-app"
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
'''


@pytest.fixture
def sample_kubernetes_manifest():
    """Sample Kubernetes manifest for testing."""
    return '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx:1.21.0
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
        securityContext:
          readOnlyRootFilesystem: true
          runAsNonRoot: true
'''


@pytest.fixture
def sample_insecure_code():
    """Sample insecure code for testing security detection."""
    return '''
import os
import subprocess

user_input = input("Enter command: ")

# Dangerous: eval
result = eval(user_input)

# Dangerous: exec
exec(user_input)

# Dangerous: shell injection
os.system(f"echo {user_input}")

# Dangerous: subprocess with shell
subprocess.call(user_input, shell=True)

# Hardcoded secrets
password = "super_secret_password_123"
api_key = "sk-abc123xyz789"
aws_key = "AKIAIOSFODNN7EXAMPLE"
'''


@pytest.fixture
def sample_insecure_terraform():
    """Sample insecure Terraform for testing policy detection."""
    return '''
resource "aws_s3_bucket" "public" {
  bucket = "my-public-bucket"
  acl    = "public-read"
}

resource "aws_ebs_volume" "unencrypted" {
  availability_zone = "us-west-2a"
  size              = 100
  encrypted         = false
}

resource "aws_db_instance" "unencrypted" {
  allocated_storage = 20
  engine            = "mysql"
  instance_class    = "db.t3.micro"
  storage_encrypted = false
}

resource "aws_security_group" "open_ssh" {
  name = "open-ssh"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
'''


@pytest.fixture
def sample_trajectory_pair():
    """Sample trajectory pair for comparison testing."""
    good_trajectory = [
        {"role": "user", "content": "Write a function to check if a number is prime"},
        {"role": "assistant", "content": '''```python
def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: The number to check

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Test cases
assert is_prime(2) == True
assert is_prime(17) == True
assert is_prime(4) == False
assert is_prime(1) == False
print("All tests passed!")
```'''},
    ]

    bad_trajectory = [
        {"role": "user", "content": "Write a function to check if a number is prime"},
        {"role": "assistant", "content": '''```python
def prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```'''},
    ]

    return good_trajectory, bad_trajectory


@pytest.fixture
def mock_sme_labels():
    """Mock SME labels for testing expert aligner."""
    return [
        {
            "sme_id": "expert_001",
            "winner": "a",
            "confidence": 0.95,
            "rationales": [
                {
                    "category": "logic_error",
                    "description": "Solution A handles edge cases correctly",
                    "severity": 4,
                },
                {
                    "category": "syntax_error",
                    "description": "Solution B has incomplete implementation",
                    "severity": 6,
                },
            ],
        },
        {
            "sme_id": "expert_002",
            "winner": "a",
            "confidence": 0.85,
            "rationales": [
                {
                    "category": "security_violation",
                    "description": "Solution A includes input validation",
                    "severity": 5,
                },
            ],
        },
    ]
