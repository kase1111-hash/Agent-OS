# Licensing Strategy

## 0. Purpose

This document defines a **deliberate, asymmetric licensing strategy** for the learning co-worker ecosystem. The goal is to:

* Protect the **Trust Kernel** from commoditization
* Prevent extraction of value without responsibility
* Allow adoption, experimentation, and contribution
* Preserve the owner-sovereignty doctrine legally, not just philosophically

Licensing is treated as a **security boundary**, not an afterthought.

---

## 1. Core Principle

> Anything that can store, constrain, or value human cognition **must not be permissively exploitable**.

Convenience licenses optimize for scale.
Trust licenses optimize for **survivability**.

---

## 2. License Layering Model

Different parts of the ecosystem require **different licenses**.

| Layer        | Examples                                          | License Goal                |
| ------------ | ------------------------------------------------- | --------------------------- |
| Trust Kernel | Boundary Daemon, Memory Vault, Learning Contracts | Prevent misuse              |
| Accounting   | Value Ledger, IntentLog                           | Preserve attribution        |
| Cognition    | synth-mind                                        | Allow research, limit abuse |
| Integration  | Agent-OS glue                                     | Encourage adoption          |
| Language     | NatLangChain docs                                 | Max readability             |

---

## 3. Recommended Licenses by Component

### 3.1 Boundary Daemon

**License:** GPLv3 or AGPLv3

**Reasoning:**

* Any modification must be disclosed
* Prevents proprietary weakening of boundaries
* Discourages embedding into closed surveillance products

Boundary enforcement must remain inspectable.

---

### 3.2 Memory Vault

**License:** GPLv3

**Reasoning:**

* Strong copyleft protects encryption logic
* Forces improvements back upstream
* Prevents silent weakening of security

---

### 3.3 Learning Contracts

**License:** GPLv3

**Reasoning:**

* Consent logic must remain auditable
* Prevents hidden expansion of learning rights

---

### 3.4 Value Ledger

**License:** LGPLv3

**Reasoning:**

* Allows linking from proprietary systems
* Prevents modification without disclosure
* Encourages adoption while preserving integrity

---

### 3.5 IntentLog

**License:** Apache 2.0 + Attribution Clause

**Reasoning:**

* Encourages widespread use
* Attribution preserves credit for effort doctrine

Optional: add NOTICE file reinforcing philosophical constraints.

---

### 3.6 synth-mind

**License:** Dual-license

* Non-commercial research: Apache 2.0
* Commercial use: Custom Trust License

**Reasoning:**

* Prevents extractive commercialization
* Allows academic exploration

---

### 3.7 Agent-OS

**License:** MIT or Apache 2.0

**Reasoning:**

* Low friction adoption
* Value is in enforcement layers, not glue

---

### 3.8 NatLangChain

**License:** Creative Commons BY-SA (docs) + Apache 2.0 (code)

**Reasoning:**

* Encourages reuse of language constructs
* Prevents enclosure of shared semantics

---

## 4. Custom Trust License (Key Asset)

For components like **synth-mind commercial use**, define a custom license requiring:

* Explicit prohibition of surveillance use
* Prohibition of training on user data without consent
* Mandatory disclosure of boundary modifications
* Owner sovereignty clause

This license should be short, human-readable, and values-forward.

---

## 5. Defensive Clauses to Include Everywhere

### 5.1 Non-Removal Clause

Trust boundary checks may not be removed or bypassed.

### 5.2 No Silent Telemetry

Any telemetry must be explicit and opt-in.

### 5.3 No Default Network Dependence

Offline operation must remain possible.

---

## 6. Contributor License Agreement (CLA)

Require a CLA that:

* Assigns no ownership to contributors
* Affirms alignment with owner sovereignty
* Prohibits backdoored contributions

---

## 7. Trademark Strategy

Register names such as:

* Trust Kernel
* Memory Vault
* Boundary Daemon

Use trademarks to prevent misuse even where licenses fail.

---

## 8. Failure Modes & Legal Threats

| Risk                      | Mitigation     |
| ------------------------- | -------------- |
| Fork strips safeguards    | Copyleft       |
| Cloud provider repackages | AGPL           |
| Surveillance misuse       | Custom clauses |
| Community fragmentation   | Clear doctrine |

---

## 9. What You Should NOT Do

* Do not use a single license everywhere
* Do not use pure MIT across trust layers
* Do not rely on ethics statements alone

---

## 10. Final Constraint

> If someone wants the value of this system without its responsibilities,
> the license should make that uncomfortable.

Licensing is the last boundary.
Treat it accordingly.
