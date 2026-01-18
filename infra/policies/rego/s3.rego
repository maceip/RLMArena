# S3 Bucket Security Policies
package vaas.s3

import future.keywords.in
import future.keywords.if

# Deny public ACLs
deny[msg] if {
    resource := input.resource.aws_s3_bucket[name]
    resource.acl == "public-read"
    msg := sprintf("S3 bucket '%s' has public-read ACL", [name])
}

deny[msg] if {
    resource := input.resource.aws_s3_bucket[name]
    resource.acl == "public-read-write"
    msg := sprintf("S3 bucket '%s' has public-read-write ACL", [name])
}

# Require encryption
deny[msg] if {
    resource := input.resource.aws_s3_bucket[name]
    not has_encryption(resource)
    msg := sprintf("S3 bucket '%s' does not have encryption enabled", [name])
}

has_encryption(bucket) if {
    bucket.server_side_encryption_configuration
}

# Require versioning
warn[msg] if {
    resource := input.resource.aws_s3_bucket[name]
    not has_versioning(resource)
    msg := sprintf("S3 bucket '%s' does not have versioning enabled", [name])
}

has_versioning(bucket) if {
    bucket.versioning.enabled == true
}

# Require tags
deny[msg] if {
    resource := input.resource.aws_s3_bucket[name]
    not has_required_tags(resource)
    msg := sprintf("S3 bucket '%s' missing required tags (Environment, Owner)", [name])
}

has_required_tags(bucket) if {
    bucket.tags.Environment
    bucket.tags.Owner
}
