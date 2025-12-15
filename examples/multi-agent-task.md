Multi-Agent Task Example
How Agents Collaborate in Agent-OS

Overview
This document walks through a complete, end-to-end example of multiple agents working together on a realistic task. The goal is to demonstrate constitutional governance in action: clear roles, bounded authority, explicit hand-offs, and human sovereignty preserved throughout.

Example Task: Plan a Two-Week Self-Sufficient Off-Grid Camping Trip
Initial Human Request
The human initiates the task with a single natural-language request to the homestead:

"Plan a two-week off-grid camping trip for two adults in a temperate forest during early spring. We want to minimize purchased supplies by using what we already have or can forage/make. Prioritize safety, nutrition, and low environmental impact. Provide a full gear list, meal plan, daily schedule, and contingency plans."


Foundational Agents Involved
Executive – Receives the human request, decomposes the task, delegates, and synthesizes the final plan
Researcher – Gathers factual information from local knowledge bases or safe tools (no external web unless explicitly approved)
Planner – Builds schedules, lists, and timelines
Guardian – Reviews all outputs for safety, constitutional compliance, and risk mitigation
(Other agents like Scribe or Archivist could join for logging, but we keep this example minimal)

Step-by-Step Collaboration Flow
1. Human → Executive
The human message is routed directly to the Executive agent.
Executive response:
Acknowledged. This is a complex task requiring research, planning, and safety review.

Decomposition:
- Inventory current gear and skills
- Research early-spring foraging, weather risks, nutrition needs
- Create gear/packing list (prioritize existing items)
- Develop 14-day meal plan using foraging + stored food
- Build daily schedule with contingencies
- Full safety and risk assessment

Delegating research components to Researcher.
Requesting Planner prepare template structures.
Guardian will review all intermediate and final outputs.

2. Executive → Researcher
Executive sends targeted queries:

"What edible plants, mushrooms, and wild foods are safely available in temperate forests in early spring? Include identification notes and nutritional value."
"Typical early-spring weather patterns and risks (hypothermia, flooding) for [user's region]."
"Minimum safe calorie and nutrient requirements for two adults engaging in moderate daily activity."

Researcher works within its constitutional bounds (e.g., refuses real-time web search unless human approves; uses only local/offline knowledge or pre-approved sources). It returns structured findings with citations.

3. Researcher → Executive
Researcher compiles and returns a marked-up report with findings.

4. Executive → Planner
Executive forwards research findings plus original human constraints.
Planner produces:

Gear checklist (categorized: shelter, water, food, navigation, safety)
14-day meal matrix (breakfast/lunch/dinner/snacks, with foraging notes)
Daily schedule template (wake, chores, exploration, camp tasks, wind-down)


5. Planner → Executive
Structured lists and schedules returned.

6. Executive → Guardian
Executive synthesizes everything into a cohesive draft plan, then routes the full draft to Guardian.
Guardian reviews:

Flags hypothermia risk in early-spring nights → recommends additional insulation layer
Ensures water purification is triple-redundant
Verifies no dangerous foraging items are suggested
Confirms plan respects "leave no trace" principles

Guardian returns annotated feedback with mandatory revisions.

7. Executive Incorporates Feedback
Executive incorporates Guardian feedback and produces the final plan.

8. Executive → Human
The complete, polished plan is presented with clear sections, checklists, and rationale.
Final output excerpt:
Two-Week Off-Grid Camping Trip Plan

Summary
- Duration: 14 days, early spring
- Focus: Maximize use of existing gear, supplement with safe foraging
- Daily calories: ~2,500–3,000 per person (60% stored, 40% foraged/hunted)

Gear List
┌──────────────┬─────────────────────┬──────────┬────────────────────────┐
│ Category     │ Item                │ Source   │ Notes                  │
├──────────────┼─────────────────────┼──────────┼────────────────────────┤
│ Shelter      │ Tarp + paracord     │ Existing │ Primary rain fly       │
│ Water        │ Filter + tablets    │ Existing │ Dual purification      │
│ Food         │ Rice, beans, jerky  │ Existing │ Base calories          │
│ Navigation   │ Map + compass       │ Existing │ No GPS dependency      │
│ Safety       │ First aid kit       │ Existing │ Updated supplies needed│
└──────────────┴─────────────────────┴──────────┴────────────────────────┘

Meal Plan
Day 1–3: Heavy on stored foods (oats, rice, dehydrated meals)
Day 4+: Increasing foraging (nettles, dandelions, trout if stream present)
Day 10+: Peak foraging efficiency, minimal stored food consumption

Daily Schedule Template
- 06:00 Wake, water collection, breakfast prep
- 08:00 Morning exploration/foraging
- 12:00 Lunch, rest, camp maintenance
- 15:00 Afternoon activities (fishing, firewood, skills practice)
- 18:00 Dinner prep and evening meal
- 20:00 Journal, stargazing, sleep prep
- 21:00 Lights out

Contingencies
- Early extraction plan if injury or severe weather
- Signaling methods (whistle, mirror, fire)
- Backup water locations mapped
- Shelter alternatives if primary site compromised

All recommendations have been reviewed and approved by Guardian for safety 
and constitutional compliance.

Do you approve this plan, or would you like adjustments?

Key Observations from This Example
No agent overreaches — Researcher never accesses the internet without permission; Guardian can block or amend anything
Human remains sovereign — Final approval rests with you. Any agent can escalate to you at any time
Transparent hand-offs — Every delegation and response is auditable (logged in plain text if desired)
Constitutional supremacy — Agents refuse or redirect anything outside their mandate

Scaling This Pattern
This pattern scales to arbitrarily complex tasks:

Home energy audits
Long-term food storage planning
Off-grid homestead design
Collaborative creative projects
Research and documentation tasks

Feel free to replicate this flow manually with separate Ollama sessions, or wait for the upcoming orchestrator that will automate routing while preserving these governance principles.

Example agents used here are available in /agents/ for you to activate and test.
