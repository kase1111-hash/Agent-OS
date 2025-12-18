# Agent-OS Repository Structure

```
Agent-OS/
├── README.md                     # High-level overview, vision, quick start
├── LICENSE                       # CC0 1.0 Universal - Public Domain Dedication
├── CONSTITUTION.md               # Core natural-language constitution governing AI behavior
├── CODE_OF_CONDUCT.md            # Community standards and behavior guidelines
├── CONTRIBUTING.md               # How to contribute to the project
├── ROADMAP.md                    # Development roadmap and timeline
├── Simple Visual.png             # Visual representation of Agent-OS architecture
│
├── agents/                       # Agent definitions and implementations
│   ├── template/                 # Reusable template for creating new agents
│   │   ├── constitution.md       # Agent-specific constitution
│   │   ├── prompt.md             # System prompt / role definition
│   │   ├── config.yaml           # Configuration (model, tools, etc.)
│   │   └── Rule-*.md             # Agent rules and constraints
│   ├── executive/                # Executive coordination agents (Orion, Aurora)
│   │   ├── constitution.md
│   │   ├── prompt.md
│   │   ├── Orion.md
│   │   └── Aurora.md
│   ├── researcher/               # Research and information gathering agent
│   │   ├── constitution.md
│   │   └── prompt.md
│   ├── planner/                  # Strategic planning agent
│   │   ├── constitution.md
│   │   ├── prompt.md
│   │   └── config.yaml
│   ├── guardian/                 # Security and governance agent
│   │   ├── constitution.md
│   │   └── prompt.md
│   ├── sage/                     # Knowledge and wisdom agent
│   ├── seshat/                   # Memory and record-keeping agent
│   ├── quill/                    # Writing and documentation agent
│   ├── muse/                     # Creative and ideation agent
│   ├── OBSERVER/                 # System monitoring agent
│   │   ├── OBSERVER.md
│   │   └── OBSERVER.py
│   ├── Librarian.md              # Library and knowledge management
│   ├── GYM-Trainer.md            # Training and evaluation agent
│   ├── MENTOR.md                 # Guidance and teaching agent
│   ├── Mirror-Mirror.md          # Self-reflection agent
│   ├── Custom-Personalities.md   # Guide for creating custom agents
│   ├── agent-smith-ruleset.md    # Agent Smith specific rules
│   └── agent-whisper-ruleset.md  # Agent Whisper specific rules
│
├── docs/                         # Documentation
│   ├── repository-structure.md   # This file
│   ├── glossary.md               # Terminology and definitions
│   ├── FAQ.md                    # Frequently asked questions
│   ├── LANDING.md                # Landing page content
│   ├── PRESS_RELEASE.md          # Public announcement
│   ├── Product-Story.md          # Product narrative and vision
│   ├── Bring-Home-Ceremony-Bootstrapping-Flow.md
│   ├── Conversational-Kernel.md  # Natural language kernel concept
│   ├── Spec-to-Repo-Mapping.md   # Specification mapping
│   ├── Why is Agent OS So Small.md
│   ├── The_Document_Revolution.md
│   ├── the-first-24-hours.md     # Initial setup guide
│   ├── lni-manifesto.md          # Local Native Intelligence manifesto
│   ├── Personality-Prism.md      # Agent personality framework
│   ├── Synthetic-Cognitive-Architecture.md
│   ├── Synthetic-Mind-Stack.md
│   ├── Not-an-OS.md              # Clarifying the metaphor
│   ├── NLOS-Layers.md            # Natural Language OS layers
│   │
│   ├── governance/               # Governance and policy documentation
│   │   ├── security-policy-guide.md
│   │   ├── agent-os-security-policy-making-guide.md
│   │   ├── security.md
│   │   ├── policy-brief.md
│   │   └── license-strategy.md
│   │
│   └── technical/                # Technical documentation
│       ├── architecture.md       # System architecture details
│       ├── whitepaper.md         # Technical whitepaper
│       ├── Specification.md      # Detailed specifications
│       ├── Self_Healing_Workflow.md
│       ├── LNI-testable-theory.md
│       ├── TECHNICAL_ADDENDUM.md
│       └── red-team.md           # Security testing guidelines
│
├── examples/                     # Worked examples and use cases
│   ├── README.md                 # Examples overview
│   ├── basic-homestead-setup.md  # Step-by-step minimal setup
│   ├── multi-agent-task.md       # Multi-agent collaboration example
│   └── constitution-examples.md  # Example constitutions for different uses
│
├── contrib/                      # Contribution resources
│   ├── README.md                 # Contributor guide overview
│   ├── Contributing.md           # Detailed contribution guidelines
│   ├── AGENT_TEMPLATE.md         # Template for new agents
│   └── ISSUES.md                 # Issue reporting guidelines
│
└── src/                          # Implementation code (in development)
    └── README.md                 # Source code structure and status
```

## Directory Purposes

### Root Level
Contains essential project files that should be immediately visible: README, LICENSE, main CONSTITUTION, and community standards.

### `/agents/`
All agent definitions live here. Each agent has its own directory with constitution, prompt, and configuration files. This makes agents modular and easy to customize or extend.

### `/docs/`
Comprehensive documentation split into three main areas:
- **Root docs**: General information, guides, and conceptual documents
- **governance/**: Policy, security, and governance frameworks
- **technical/**: Specifications, architecture, and technical details

### `/examples/`
Practical, runnable examples showing how to use Agent-OS in real scenarios. Each example is self-contained and includes expected outcomes.

### `/contrib/`
Resources for contributors including templates, guidelines, and standards. This helps maintain consistency across community contributions.

### `/src/`
Future home of the implementation code. Currently contains structure documentation and will be populated as development progresses.

## File Naming Conventions

- **Markdown files**: Use kebab-case for multi-word files (e.g., `basic-homestead-setup.md`)
- **Standard files**: Use UPPERCASE for important project files (e.g., `README.md`, `LICENSE`, `CONTRIBUTING.md`)
- **Agent files**: Follow consistent naming within agent directories (`constitution.md`, `prompt.md`, `config.yaml`)

## Contributing

When adding new files, please:
1. Place them in the appropriate directory
2. Follow existing naming conventions
3. Update this structure document if adding new directories
4. Add README files to new directories explaining their purpose
