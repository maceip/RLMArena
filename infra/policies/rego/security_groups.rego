# Security Group Policies
package vaas.security_groups

import future.keywords.in
import future.keywords.if

# Deny unrestricted SSH access
deny[msg] if {
    resource := input.resource.aws_security_group[name]
    ingress := resource.ingress[_]
    ingress.from_port <= 22
    ingress.to_port >= 22
    "0.0.0.0/0" in ingress.cidr_blocks
    msg := sprintf("Security group '%s' allows SSH from 0.0.0.0/0", [name])
}

# Deny unrestricted RDP access
deny[msg] if {
    resource := input.resource.aws_security_group[name]
    ingress := resource.ingress[_]
    ingress.from_port <= 3389
    ingress.to_port >= 3389
    "0.0.0.0/0" in ingress.cidr_blocks
    msg := sprintf("Security group '%s' allows RDP from 0.0.0.0/0", [name])
}

# Warn on wide port ranges
warn[msg] if {
    resource := input.resource.aws_security_group[name]
    ingress := resource.ingress[_]
    port_range := ingress.to_port - ingress.from_port
    port_range > 100
    msg := sprintf("Security group '%s' has wide port range (%d ports)", [name, port_range])
}
