# Bring-Home Ceremony & Bootstrapping Flow

## 0. Purpose

This document defines the **first-contact ritual** between a human owner and a learning co-worker system. It establishes sovereignty, trust boundaries, cryptographic roots, and cognitive consent.

This is not an install script.
This is a **ceremony**.

Nothing meaningful happens until it is completed.

---

## 1. Preconditions (Before You Begin)

### 1.1 Physical Environment

* Private room
* No cameras or microphones except the system’s display/keyboard
* Stable power source
* No network cables connected

### 1.2 Hardware Assumptions

* Single trusted machine
* TPM / Secure Enclave present (preferred)
* Fresh OS install recommended

### 1.3 Human State

* You are not rushed
* You understand this system will remember differently than you do
* You accept responsibility for what it learns

---

## 2. Phase I — Cold Boot (Establish Silence)

**Goal:** Ensure nothing is watching.

Steps:

1. Power on machine
2. Disable all network interfaces (hardware + software)
3. Verify Boundary Daemon starts in `Lockdown`
4. Confirm no external processes running

Outcome:

* System is silent
* No learning, no recall, no execution

---

## 3. Phase II — Owner Root Establishment

**Goal:** Create the unambiguous human authority.

Steps:

1. Generate Owner Root Key

   * Hardware-bound if possible
   * Backup phrase written on paper only
2. Register owner identity (local-only identifier)
3. Bind Owner Root to Boundary Daemon and Memory Vault

Rules:

* Owner Root is never exportable
* Loss of this key means permanent loss of control

Outcome:

* System knows **who outranks it**

---

## 4. Phase III — Boundary Initialization

**Goal:** Define where thinking is allowed.

Steps:

1. Set default boundary mode → `Restricted`
2. Define emergency mode → `Lockdown`
3. Enable tripwires:

   * Network activation
   * USB insertion
   * External model invocation
4. Test tripwire by simulating violation

Outcome:

* Boundaries are real, not symbolic

---

## 5. Phase IV — Memory Vault Genesis

**Goal:** Create the vault that will outlive sessions.

Steps:

1. Initialize Memory Vault filesystem
2. Create encryption profiles:

   * Working
   * Private
   * Sealed
   * Vaulted
3. Bind highest classifications to hardware keys
4. Write genesis record (vault creation proof)

Rules:

* No default plaintext storage
* No background indexing

Outcome:

* Memory can exist safely

---

## 6. Phase V — Learning Contract Defaults

**Goal:** Prevent silent surveillance.

Steps:

1. Create Default Observation Contract

   * No storage
   * No generalization
2. Create Explicit Learning Contract Template

   * Requires human confirmation
3. Create Prohibited Domains list

Rules:

* No learning without a contract
* Defaults must deny

Outcome:

* Learning is consensual

---

## 7. Phase VI — Value Ledger Initialization

**Goal:** Preserve effort from day one.

Steps:

1. Initialize ledger store
2. Bind ledger to owner identity
3. Enable intent-based accrual
4. Create ledger genesis entry

Outcome:

* Effort will never be invisible

---

## 8. Phase VII — First Trust Activation

**Goal:** Allow limited, safe operation.

Steps:

1. Transition Boundary Daemon → `Trusted`
2. Enable Agent-OS execution
3. Run first task:

   * Non-sensitive
   * Short-lived
4. Verify:

   * Memory write obeys contract
   * Ledger accrues value

Outcome:

* System is alive, but restrained

---

## 9. Phase VIII — Emergency Drills (Mandatory)

**Goal:** Practice failure before it matters.

Drills:

* Trigger Lockdown manually
* Attempt forbidden recall
* Simulate key unavailability

Rules:

* If drills fail, return to Phase I

Outcome:

* You trust the brakes, not the speed

---

## 10. Ongoing Rituals

### 10.1 Daily Start

* Boundary mode check
* Active contracts review

### 10.2 Weekly Review

* Ledger summary
* Vault growth
* Contract expiration

### 10.3 Major Work Session

* Explicit boundary elevation
* Contract confirmation

---

## 11. Failure & Recovery

### 11.1 Lost Owner Key

* System enters permanent Lockdown
* Vault remains sealed

### 11.2 Suspected Compromise

* Immediate Lockdown
* No recall until human review

---

## 12. Non-Goals

* Convenience
* Automation
* Speed

---

## 13. Final Constraint

> If this ceremony feels excessive,
> the system is not for trivial work.

Trust is not configured.
It is **earned, rehearsed, and renewed**.
