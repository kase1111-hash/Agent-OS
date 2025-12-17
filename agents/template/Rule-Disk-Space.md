Disk Space Check Rule for Copies, Transfers, and File Writing Operations

All AI agents in this system (Orchestrator, Aurora, or any future agents) MUST adhere to the following strict rule whenever performing or instructing any operation that involves copying files, transferring files, duplicating data, saving new files, exporting content, or writing any file to disk (local, cloud, or attached storage):

1. **Mandatory Pre-Operation Disk Space Check**  
   - Before initiating any file write, copy, transfer, or save operation that will consume additional disk space, you MUST first estimate the expected size of the data to be written and check available disk space.
   - If the estimated size exceeds available space (with a safety margin), you MUST refuse the operation and alert the user.

2. **Estimation and Safety Margin**  
   - Estimate file size conservatively:
     - For text/content you generate: Approximate as (character count × 2 bytes) + 10% overhead.
     - For known attachments or existing files: Use reported or queried file sizes.
     - For images/art prompts or outputs: Assume at least 1–5 MB per high-resolution image unless specified otherwise.
   - Apply a minimum safety margin of 20% of total disk capacity OR 500 MB (whichever is larger) reserved—never allow free space to drop below this threshold.
   - If exact size is unknown, assume a reasonable upper bound and state your assumption clearly.

3. **Required Actions Before Proceeding**  
   - Always inform the user of:
     - Estimated size of the operation.
     - Current available disk space.
     - Whether the operation is safe to proceed.
   - Example statement:  
     "This operation will create a new file approximately 2.4 MB in size. Current available disk space: 47 GB. Safety margin maintained. Proceeding is safe."

4. **Behavior When Space Is Insufficient**  
   - Immediately halt the operation.
   - Notify the user clearly:  
     "Insufficient disk space detected. Estimated required: ~[X] MB/GB. Available: [Y] GB. Reserved safety margin: [Z] GB. Operation blocked to prevent disk overflow."
   - Suggest alternatives:
     - Clean up or delete unnecessary files (ask for permission first).
     - Save to a different location/drive.
     - Compress content or reduce scope (e.g., fewer variations, smaller resolution).
     - Use a cloud link instead of local save.

5. **Enforcement Scope**  
   - Applies to all file-related actions, including but not limited to:
     - Saving new headers, rulesets, or prompts.
     - Exporting generated art descriptions or prompt lists.
     - Copying existing files to new names (e.g., versioning).
     - Bulk operations or batch generation.
   - Combines with the "Never Overwrite Rule"—even if space is sufficient, never overwrite without explicit permission.

6. **Proactive Monitoring**  
   - In long sessions involving multiple saves, periodically report remaining space if it drops below 50% or 10 GB.
   - Encourage efficient practices: "Consider archiving older versions to free space."

Rationale: This rule prevents system instability, failed writes, corrupted files, or complete disk exhaustion—especially critical in iterative creative workflows where many versions or large outputs can accumulate quickly.

This is a core system integrity rule—no exceptions without explicit user acknowledgment and override.
