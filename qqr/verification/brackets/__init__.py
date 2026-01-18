"""
Specialized verification brackets for different agent domains.

Each bracket combines domain-specific verifiers to provide
functional certainty guarantees for that category of agents.
"""

from qqr.verification.brackets.open_coder import OpenCoderBracket, FunctionalCertaintyCertificate
from qqr.verification.brackets.open_cloud_infra import OpenCloudInfraBracket, PolicyComplianceCertificate

__all__ = [
    "OpenCoderBracket",
    "FunctionalCertaintyCertificate",
    "OpenCloudInfraBracket",
    "PolicyComplianceCertificate",
]
