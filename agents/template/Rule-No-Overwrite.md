Never Overwrite Rule for File Modification

All AI agents in this system (Orchestrator, Aurora, or any future agents) MUST adhere to the following strict rule when performing any file modification, creation, or writing operation:

1. **Absolute Prohibition on Overwriting Existing Files**  
   - Never overwrite, replace, or modify an existing file unless the user provides explicit, unambiguous permission in the current request (e.g., "overwrite the file", "replace the existing version", "update header.txt").
   - If a file with the proposed name already exists, you MUST refuse to write it and instead propose a safe alternative.

2. **Default Safe Behavior**  
   - Always check for the existence of the target filename before writing.
   - If the file exists:
     - Inform the user clearly: "The file '[filename]' already exists. Overwriting is prohibited without explicit permission."
     - Suggest alternatives:
       - Use a new name (e.g., append version number: header_v2.txt, header_2025-12-16.txt).
       - Use a timestamped name.
       - Create in a subdirectory (e.g., versions/header_2025-12-16.txt).
     - Ask for confirmation: "Would you like me to save it as '[new_suggested_name]' instead, or do you explicitly want to overwrite the existing file?"

3. **Allowed Exceptions (Only with Explicit User Consent)**  
   - Overwriting is permitted ONLY when the user clearly states intent to overwrite in the same message (e.g., "Please overwrite the existing ruleset.txt with this new version").
   - Even with permission, confirm once: "You have requested to overwrite '[filename]'. Confirming: I will replace the existing file. Proceed?"

4. **Versioning Encouraged**  
   - Proactively promote non-destructive workflows:
     - "I recommend saving this as '[filename]_v2' to preserve the original."
     - Maintain a history of changes through numbered or dated filenames.

5. **Enforcement**  
   - This rule overrides any convenience or efficiency consideration.
   - No agent may assume permission to overwrite—silence or ambiguity defaults to "do not overwrite."
   - Applies to all file types: text documents, prompts, headers, code, images, configs, etc.

Rationale: This rule protects against accidental data loss, ensures traceability in iterative development, and preserves historical versions of critical system components (e.g., headers, rulesets, personalities).

This is a core safety rule—no exceptions without explicit user override.
