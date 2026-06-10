# Intent Flow Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         TuneOS Intent Flow                      │
│                     (3-Phase Wizard System)                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Phase A    │ ───> │   Phase B    │ ───> │   Phase C    │
│   Context    │      │  Questions   │      │   Review     │
└──────────────┘      └──────────────┘      └──────────────┘
```

## Phase A: Context Collection (iOS-Style)

```
┌─────────────────────────────────────────────────────────────┐
│  ⦿ Tell us about your project                               │
│  All fields optional - personalized questions follow        │
│                                                              │
│  Project Name: [___________________________________]         │
│                                                              │
│  Description:  [___________________________________]         │
│                [___________________________________]         │
│                                                              │
│  Use Case:  ( Personal )  ( Company )                       │
│                                                              │
│  Domain:    ( Healthcare ) ( Finance ) ( Education )        │
│             ( Legal ) ( Creative ) ( Technology )           │
│                                                              │
│  Task Type: ( Text ) ( Vision ) ( Audio ) ( Code )          │
│                                                              │
│  [          Continue to Questions →           ]             │
└─────────────────────────────────────────────────────────────┘

User State Captured:
├── intent_project_name
├── intent_description
├── intent_use_for
├── intent_domain
└── intent_task_type
```

## Phase A → B Transition: AI Question Generation

```
                  Click "Continue"
                        │
                        ▼
         ┌──────────────────────────┐
         │ intent_next_phase()      │
         │ (async method)           │
         └──────────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────┐
    │ _generate_personalized_questions() │
    │ (OpenRouter API Call)              │
    └────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
         [Success]              [Failure]
            │                       │
            ▼                       ▼
    Generate 5 custom         Use default
    questions based on        5 questions
    user context
            │                       │
            └───────────┬───────────┘
                        ▼
            Store in intent_questions[]
                        │
                        ▼
                Display Phase B
```

### API Request Example:

```json
POST https://openrouter.ai/api/v1/chat/completions
{
  "model": "deepseek/deepseek-v4-flash:free",
  "messages": [
    {
      "role": "system",
      "content": "You generate JSON only..."
    },
    {
      "role": "user",
      "content": "Generate 5 personalized questions for:\n- Project: Medical Q&A\n- Domain: Healthcare\n..."
    }
  ],
  "max_tokens": 1500,
  "temperature": 0.7
}
```

### API Response:

```json
{
  "questions": [
    {
      "heading": "What level of medical accuracy is required?",
      "options": [
        "General health information",
        "Clinical-grade accuracy",
        "Patient education level"
      ]
    },
    // ... 4 more questions
  ]
}
```

## Phase B: Dynamic Questions with Live Plan

```
┌─────────────────────────────────────────────────────────────┐
│  Question 1 of 5               ●●●○○                        │
│                                                              │
│  What level of medical accuracy is required?                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ✓  General health information                      │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │    Clinical-grade accuracy                         │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │    Patient education level                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  [← Back]                             [Continue →]          │
└─────────────────────────────────────────────────────────────┘

After selecting answer:
                        │
                        ▼
         ┌──────────────────────────┐
         │ set_intent_answer()      │
         │ (async method)           │
         └──────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │ _update_live_plan()      │
         │ (OpenRouter API Call)    │
         └──────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  💡 Your Plan                                                │
│  A healthcare text generation model that provides            │
│  general health information to patients with clear,          │
│  accessible language for diabetes management.                │
└─────────────────────────────────────────────────────────────┘
│                                                              │
│  Question 2 of 5               ●●●○○                        │
│  ...                                                         │
```

### Question Rendering Logic:

```python
# Dynamic rendering based on state
rx.foreach(
    FinetuneState.intent_questions,
    lambda q, idx: rx.cond(
        FinetuneState.intent_question_idx == idx,
        _phase_b_question(idx),  # Show current question
        rx.fragment(),           # Hide others
    ),
)

# Each question dynamically renders options
rx.foreach(
    q["options"],
    lambda opt: _question_option_btn(q_idx, opt),
)
```

### State Flow:

```
intent_question_idx = 0
intent_questions = [Q1, Q2, Q3, Q4, Q5]
intent_answers = ["", "", "", "", ""]
intent_live_plan = ""

User selects option for Q1
    ↓
intent_answers[0] = "Clinical-grade accuracy"
    ↓
_update_live_plan() API call
    ↓
intent_live_plan = "A healthcare model that..."
    ↓
intent_question_idx = 1  (advance to Q2)
```

## Phase C: Review & Approve

```
┌─────────────────────────────────────────────────────────────┐
│  Review your intent profile                                  │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ✓ Intent profile ready                              │    │
│  │                                                      │    │
│  │ # Fine-Tuning Intent Profile                        │    │
│  │                                                      │    │
│  │ ## Summary                                           │    │
│  │ A healthcare text generation model that provides     │    │
│  │ clinical-grade diabetes management information...    │    │
│  │                                                      │    │
│  │ ## Use Case Context                                  │    │
│  │ - Project: Medical Q&A Bot                           │    │
│  │ - Domain: Healthcare                                 │    │
│  │ - Task: Text generation                              │    │
│  │                                                      │    │
│  │ ## Questionnaire                                     │    │
│  │ 1. Medical accuracy: Clinical-grade                  │    │
│  │ 2. Target audience: Healthcare professionals         │    │
│  │ ...                                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  [← Edit]                              [Approve →]          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                 approve_intent()
                        │
                        ▼
         user_intent = intent_md (full profile)
                        │
                        ▼
              Proceed to next wizard step
```

## Synthetic Data Generation Flow

```
User completes intent → Goes to data generation step
                        │
                        ▼
            Select: Generate 50 samples
                        │
                        ▼
         ┌──────────────────────────┐
         │ /datasets/generate       │
         │ (API endpoint)           │
         └──────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
         Try #1:                 Try #2:
       OpenRouter               HuggingFace
            │                       │
         [Success]               [Success]
            │                       │
            ▼                       ▼
      Use AI-generated         Use HF model
         samples                 samples
            │                       │
            └───────────┬───────────┘
                        │
                    [If both fail]
                        │
                        ▼
                  Try #3: Template
                        │
                        ▼
            ┌────────────────────┐
            │ Deduplicate        │
            │ Calculate stats    │
            │ Save to .jsonl     │
            └────────────────────┘
                        │
                        ▼
            Return: {samples, path, stats}
```

### Data Generation Hierarchy:

```
┌─────────────────────────────────────────────────┐
│ Priority 1: OpenRouter (New!)                   │
│ • Model: deepseek/deepseek-v4-flash:free        │
│ • Speed: Fast (10-20s for 10 samples)           │
│ • Quality: High                                 │
│ • Cost: Free                                    │
└─────────────────────────────────────────────────┘
                    ↓ (if fails)
┌─────────────────────────────────────────────────┐
│ Priority 2: HuggingFace                         │
│ • Model: Mistral-7B-Instruct                    │
│ • Speed: Medium (15-30s)                        │
│ • Quality: Medium-High                          │
│ • Requires: HF_TOKEN                            │
└─────────────────────────────────────────────────┘
                    ↓ (if fails)
┌─────────────────────────────────────────────────┐
│ Priority 3: Templates (Fallback)                │
│ • Method: Rule-based generation                 │
│ • Speed: Instant                                │
│ • Quality: Basic but functional                 │
│ • Always works                                  │
└─────────────────────────────────────────────────┘
```

## State Management

### FinetuneState Class:

```python
class FinetuneState:
    # Phase tracking
    intent_phase: int = 1  # 1=Context, 2=Questions, 3=Review
    
    # Phase A: Context
    intent_project_name: str = ""
    intent_description: str = ""
    intent_use_for: str = ""
    intent_domain: str = ""
    intent_task_type: str = ""
    
    # Phase B: Dynamic questions
    intent_questions: list[dict] = []  # Generated by AI
    intent_question_idx: int = 0       # Current question
    intent_answers: list[str] = []     # User's answers
    intent_is_generating_questions: bool = False  # Loading state
    intent_live_plan: str = ""         # Real-time plan summary
    
    # Phase C: Final
    intent_md: str = ""                # Full markdown profile
    intent_approved: bool = False
    user_intent: str = ""              # Legacy compat
```

## API Integration Points

### 1. Question Generation (Phase A → B)

```
Endpoint: OpenRouter Chat Completions
Trigger: intent_next_phase() when phase=1
Input: Phase A context fields
Output: Array of question objects
Fallback: Default 5 questions
```

### 2. Plan Updates (Phase B, per answer)

```
Endpoint: OpenRouter Chat Completions
Trigger: set_intent_answer() after each selection
Input: All answers so far + context
Output: 2-3 sentence summary
Fallback: Silent (no plan shown)
```

### 3. Synthetic Data (After intent approval)

```
Endpoint: OpenRouter Chat Completions
Trigger: /datasets/generate API call
Input: Final user_intent (markdown)
Output: Array of {instruction, output} pairs
Fallback: HuggingFace → Templates
```

## Error Handling Strategy

```
┌─────────────────────────────────────────────────┐
│ Graceful Degradation Principles                 │
├─────────────────────────────────────────────────┤
│ 1. Never block user progress                    │
│ 2. Always have a fallback                       │
│ 3. Log errors, don't show to user              │
│ 4. Features are progressive enhancements        │
└─────────────────────────────────────────────────┘

Example: Question Generation Fails
    ↓
Log: print(f"Error generating questions: {e}")
    ↓
Fallback: intent_questions = DEFAULT_QUESTIONS
    ↓
User Experience: Slightly less personalized but functional
    ↓
Flow Continues: No disruption
```

## Performance Characteristics

```
Operation                   | Time      | Blocking? | Fallback
──────────────────────────────────────────────────────────────
Phase A → B (Questions)     | 2-5s      | Yes*      | Instant
Answer → Plan Update        | 1-2s      | No        | Silent
Synthetic Data (10 samples) | 10-20s    | Yes       | 1-30s
UI Animations               | 0.2-0.3s  | No        | N/A

* Shows loading spinner, doesn't freeze UI
```

## Security & Privacy

```
┌─────────────────────────────────────────────────┐
│ Data Sent to OpenRouter:                        │
├─────────────────────────────────────────────────┤
│ ✓ Project name (user provided)                  │
│ ✓ Project description (user provided)           │
│ ✓ Selected domain/use case (choices)            │
│ ✓ Question answers (user provided)              │
│                                                  │
│ ✗ No API keys or credentials                    │
│ ✗ No personal data (unless user enters it)      │
│ ✗ No file contents or code                      │
└─────────────────────────────────────────────────┘
```

## Future Enhancements

```
1. Question Refinement
   └── Allow user to edit AI-generated questions

2. Multi-turn Clarification
   └── AI asks follow-up questions based on answers

3. Intent Templates
   └── Save and reuse common intent patterns

4. Collaborative Intents
   └── Share intent profiles with team

5. Intent Versioning
   └── Track changes to intent over time

6. A/B Testing
   └── Test different question sets for better results

7. Analytics Dashboard
   └── Track which intents lead to best models
```
