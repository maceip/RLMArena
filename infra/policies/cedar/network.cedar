// VaaS Network Policies - Cedar
// Controls what network actions agents can perform

// Default deny all
forbid (
    principal,
    action,
    resource
);

// Allow connections to approved APIs
permit (
    principal is Agent,
    action == Action::"connect",
    resource is Host
) when {
    resource.domain in [
        "api.github.com",
        "api.openai.com",
        "api.anthropic.com",
        "huggingface.co",
        "cdn-lfs.huggingface.co"
    ]
};

// Allow HTTPS to external services
permit (
    principal is Agent,
    action == Action::"connect",
    resource is Host
) when {
    resource.port == 443 &&
    resource.protocol == "https"
};

// Deny connections to localhost/internal
forbid (
    principal is Agent,
    action == Action::"connect",
    resource is Host
) when {
    resource.domain in [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254.169.254",
        "metadata.google.internal"
    ]
};

// Deny connections to private IP ranges
forbid (
    principal is Agent,
    action == Action::"connect",
    resource is Host
) when {
    resource.ip_private == true
};

// Allow DNS resolution
permit (
    principal is Agent,
    action == Action::"resolve",
    resource is Domain
);

// Audit all denied connections
@advice("log_denied_connection")
forbid (
    principal is Agent,
    action == Action::"connect",
    resource
);
