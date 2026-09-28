"""Save an IBM Quantum account from environment variables (never hard-code credentials).

  export IBM_QUANTUM_TOKEN=...          # API key
  export IBM_QUANTUM_CRN=...            # optional: instance CRN to use as default
  uv run python scripts/ibm_login.py

scripts/qbio_runtime.get_service() then pins the instance by name or CRN for every call
($QBIO_IBM_INSTANCE), so a default account in another region cannot mislead job retrieval.
"""
import os

from qiskit_ibm_runtime import QiskitRuntimeService

token = os.environ["IBM_QUANTUM_TOKEN"]
crn = os.environ.get("IBM_QUANTUM_CRN")
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=token, instance=crn, overwrite=True)
print("IBM Quantum account saved" + (" (default instance set)" if crn else ""))
