# Spec-to-Repo Mapping (Concrete File Layout)

This document maps the **Trust Kernel specs** to concrete files and directories across your existing repositories, with minimal restructuring.

Repos assumed:

* Agent-OS
* synth-mind
* IntentLog
* NatLangChain

New lightweight repos/modules are introduced only where unavoidable.

---

## 1. Agent-OS (Execution & Enforcement)

**Role:** System spine, mandatory enforcement hooks.

### Add / Modify

```
agent-os/
├─ core/
│  ├─ boundary_client.py        # Mandatory calls to Boundary Daemon
│  ├─ contract_client.py        # Learning contract checks
│  ├─ vault_client.py           # Memory write/read requests
│  └─ ledger_client.py          # Value accrual hooks
│
├─ runtime/
│  ├─ task_executor.py          # Enforces allow/deny before execution
│  └─ tool_router.py            # IO + tool gating
│
├─ security/
│  └─ enforcement.md            # Non-bypass doctrine
└─ README.md                    # Declares Agent-OS as subordinate to Trust Kernel
```

**Key Rule:** Agent-OS never decides — it asks.

---

## 2. synth-mind (Cognition)

**Role:** Thought generation under constraint.

### Add / Modify

```
synth-mind/
├─ cognition/
│  ├─ learning_gate.py          # Contract-aware learning interface
│  ├─ abstraction_guard.py      # Prevents over-generalization
│  └─ recall_filter.py          # Least-recall enforcement
│
├─ reflection/
│  └─ self_model.md             # Explicit non-sovereignty statement
│
├─ experiments/
│  └─ sandbox_only/             # Must never touch vault
└─ README.md                    # Declares synth-mind as non-authoritative
```

**Key Rule:** synth-mind may *propose*, never persist.

---

## 3. IntentLog (Effort & Intent Capture)

**Role:** Canonical source of human intent.

### Add / Modify

```
IntentLog/
├─ schema/
│  ├─ intent.json               # Intent schema (unchanged)
│  └─ intent_value_link.json    # Maps intent → ledger entries
│
├─ hooks/
│  └─ on_intent_close.py        # Triggers value accrual
│
├─ audits/
│  └─ intent_chain.log          # Append-only intent history
└─ README.md                    # Intent outranks outcome doctrine
```

**Key Rule:** No intent → no value.

---

## 4. NatLangChain (Proof & Shared Semantics)

**Role:** Human-readable proofs and contracts.

### Add / Modify

```
NatLangChain/
├─ contracts/
│  ├─ learning_contract.nlc     # Plain-language learning contracts
│  └─ prohibited_domains.nlc
│
├─ proofs/
│  ├─ ledger_proof.nlc          # Value summaries without content
│  └─ vault_existence.nlc       # Proof of creation
│
├─ spec_bindings/
│  └─ nlc_to_json.py            # Translation layer
└─ README.md                    # States NLC is descriptive, not authoritative
```

**Key Rule:** Language describes truth; it does not enforce it.

---

## 5. NEW: boundary-daemon (Hard Trust Enforcement)

**This must be separate.** Do not embed it.

```
boundary-daemon/
├─ daemon/
│  ├─ state_monitor.py          # Network, hardware, process sensing
│  ├─ policy_engine.py          # Mode × signal × request
│  └─ tripwires.py
│
├─ api/
│  └─ boundary.sock             # Local-only command socket
│
├─ logs/
│  └─ boundary_chain.log        # Immutable event log
└─ README.md
```

---

## 6. NEW: memory-vault (Secure Storage)

```
memory-vault/
├─ store/
│  ├─ encrypted_blobs/
│  └─ index.db
│
├─ crypto/
│  ├─ key_manager.py
│  └─ profiles.json
│
├─ api/
│  └─ vault.sock
└─ README.md
```

---

## 7. NEW: value-ledger (Accounting)

```
value-ledger/
├─ ledger/
│  ├─ entries.db
│  ├─ accrual.py
│  └─ aggregation.py
│
├─ proofs/
│  └─ merkle.py
└─ README.md
```

---

## 8. NEW: learning-contracts (Consent Engine)

```
learning-contracts/
├─ contracts/
│  ├─ active/
│  ├─ expired/
│  └─ revoked/
│
├─ enforcement/
│  └─ contract_validator.py
└─ README.md
```

---

## 9. Cross-Repo Invariants (Non-Negotiable)

1. Boundary Daemon must be running or system halts
2. No component writes memory without a contract
3. Ledger records meta only — never content
4. synth-mind cannot escalate abstraction on its own
5. Owner Root outranks all automation

---

## 10. What You Can Commit Today

Minimum viable commit set:

* Add enforcement clients to Agent-OS
* Add learning gate to synth-mind
* Add intent → value hook to IntentLog
* Publish Boundary Daemon as empty scaffold

Everything else can evolve.

---

## Final Constraint

> If enforcement is optional, trust is imaginary.

This mapping makes enforcement unavoidable.
