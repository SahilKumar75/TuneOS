# UI Examples - Before & After

## Phase A: Context Collection

### Before 
```
┌─────────────────────────────────────────┐
│ Step 1 of 3 — Context filters           │
│                                          │
│ Use for?                                 │
│ [Personal] [Company]                     │
│                                          │
│ Domain?                                  │
│ [Healthcare][Finance][Education][Legal]  │
│                                          │
│ Task type?                               │
│ [Text][Vision][Audio][Code]              │
│                                          │
│ [Continue →]                             │
└─────────────────────────────────────────┘
```

### After 
```
┌──────────────────────────────────────────────────────────┐
│   Tell us about your project                            │
│ 1  All fields optional — personalized questions follow   │
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │                                                      │  │
│ │  Project Name                                        │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │ e.g., Medical Q&A Assistant                  │   │  │
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ │  Description                                         │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │ Briefly describe what your model will do...  │   │  │
│ │  │                                               │   │  │
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ │  ─────────────────────────────────────────────────  │  │
│ │                                                      │  │
│ │  Use Case                                            │  │
│ │  Who will use this model?                            │  │
│ │                                                      │  │
│ │  ╭──────────────╮  ╭──────────────────╮              │  │
│ │  │  Personal    │  │ Company product  │              │  │
│ │  ╰──────────────╯  ╰──────────────────╯              │  │
│ │                                                      │  │
│ │  ─────────────────────────────────────────────────  │  │
│ │                                                      │  │
│ │  Domain                                              │  │
│ │  What industry or field?                             │  │
│ │                                                      │  │
│ │  ╭──────────────╮ ╭─────────╮ ╭───────────╮         │  │
│ │  │ Healthcare   │ │ Finance │ │ Education │         │  │
│ │  ╰──────────────╯ ╰─────────╯ ╰───────────╯         │  │
│ │  ╭───────╮ ╭──────────╮ ╭────────────╮              │  │
│ │  │ Legal │ │ Creative │ │ Technology │              │  │
│ │  ╰───────╯ ╰──────────╯ ╰────────────╯              │  │
│ │                                                      │  │
│ │  ─────────────────────────────────────────────────  │  │
│ │                                                      │  │
│ │  Task Type                                           │  │
│ │  What kind of output?                                │  │
│ │                                                      │  │
│ │  ╭──────╮ ╭────────╮ ╭───────╮ ╭──────╮             │  │
│ │  │ Text │ │ Vision │ │ Audio │ │ Code │             │  │
│ │  ╰──────╯ ╰────────╯ ╰───────╯ ╰──────╯             │  │
│ │                                                      │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
│ ┌───────────────────────────────────────────────────┐   │
│ │    Continue to Questions  →                       │   │
│ └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘

Hover Effects:
• Chips lift up (translateY(-2px))
• Buttons show shadow
• Inputs highlight blue border on focus
```

## Loading State (Phase A → B Transition)

### New 
```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │                                                      │  │
│ │                                                     │  │
│ │                                                      │  │
│ │            Generating personalized questions...      │  │
│ │                                                      │  │
│ │               Based on your project details          │  │
│ │                                                      │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Animation:
• Spinner rotates smoothly
• Card fades in
• 2-5 second duration
```

## Phase B: Questions

### Before 
```
┌─────────────────────────────────────────┐
│ Question 1 of 5            ●●○○○         │
│                                          │
│ What is the primary goal?                │
│                                          │
│ [Answer questions / provide info]        │
│ [Generate or transform content]          │
│ [Classify, analyze, or extract]          │
│ [ Other... ]                             │
│                                          │
│ [← Back]               [Next →]          │
└─────────────────────────────────────────┘
```

### After 
```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │  Your Plan                                         │  │
│ │                                                      │  │
│ │ A healthcare text generation model that provides     │  │
│ │ clinical-grade diabetes management information to    │  │
│ │ patients and healthcare professionals.               │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │                                                      │  │
│ │  Question 2 of 5        ●●●●○                        │  │
│ │                                                      │  │
│ │  Who is the target audience?                         │  │
│ │                                                      │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │   Healthcare professionals                  │   │  │
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │    Patients and caregivers                   │   │  │
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │    General public                            │   │  │
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ │  ╭──────────────────────────────────────────────╮   │  │
│ │  │    Other...                                  │   │  │
│ │  ╰──────────────────────────────────────────────╯   │  │
│ │                                                      │  │
│ │  ┌─────────────┐                   ┌────────────┐   │  │
│ │  │ ← Back      │                   │ Continue → │   │  │
│ │  └─────────────┘                   └────────────┘   │  │
│ │                                                      │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Features:
• Live plan in amber card (updates per answer)
• Check icon on selected option
• Progress dots show position
• Smooth animations on interaction
• "Other" expands when clicked
```

### "Other" Expanded State
```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │  ...previous options...                              │  │
│ │                                                      │  │
│ │  ╭──────────────────────────────────────────────╮   │  │
│ │  │    Other...                                  │   │  │
│ │  ╰──────────────────────────────────────────────╯   │  │
│ │                                                      │  │
│ │  ┌──────────────────────────────────────────────┐   │  │
│ │  │ Medical students and residents               │   │  │ ← User typing
│ │  └──────────────────────────────────────────────┘   │  │
│ │                                                      │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Animation:
• Smooth slide down
• Input auto-focused
• Blue border glow on focus
```

## Phase C: Review

### Before 
```
┌─────────────────────────────────────────┐
│ Review your intent profile               │
│                                          │
│ ┌────────────────────────────────────┐  │
│ │ # Intent Profile                    │  │
│ │ Summary: A model for...             │  │
│ │ Domain: Healthcare                  │  │
│ │ ...                                 │  │
│ └────────────────────────────────────┘  │
│                                          │
│ [← Edit]              [Approve →]        │
└─────────────────────────────────────────┘
```

### After 
```
┌──────────────────────────────────────────────────────────┐
│  Review your intent profile                               │
│                                                           │
│  This profile guides data generation, training config,    │
│  and system prompt scaffolding.                           │
│                                                           │
│ ╭─────────────────────────────────────────────────────╮  │
│ │                                                      │  │
│ │   Intent profile ready                              │  │
│ │                                                      │  │
│ │  # Fine-Tuning Intent Profile                        │  │
│ │                                                      │  │
│ │  ## Summary                                           │  │
│ │  A healthcare text generation model that provides     │  │
│ │  clinical-grade diabetes management information to    │  │
│ │  patients and healthcare professionals using          │  │
│ │  structured medical records.                          │  │
│ │                                                      │  │
│ │  ## Use Case Context                                  │  │
│ │  - **Project Name:** Medical Q&A Bot                  │  │
│ │  - **Description:** Answer patient questions about    │  │
│ │    diabetes management                                │  │
│ │  - **Use for:** Personal                              │  │
│ │  - **Domain:** Healthcare                             │  │
│ │  - **Task type:** Text generation                     │  │
│ │                                                      │  │
│ │  ## Questionnaire                                     │  │
│ │  1. **Medical accuracy:** Clinical-grade              │  │
│ │  2. **Target audience:** Healthcare professionals     │  │
│ │  3. **Input format:** Structured medical records      │  │
│ │  4. **Tone:** Formal and precise                      │  │
│ │  5. **Success metric:** Accuracy / correctness        │  │
│ │                                                      │  │
│ ╰─────────────────────────────────────────────────────╯  │
│                                                           │
│ ┌─────────────┐                     ┌─────────────────┐  │
│ │  ← Edit     │                     │  Approve →      │  │
│ └─────────────┘                     └─────────────────┘  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Features:
• Check icon confirmation
• Full markdown rendering
• Scrollable content area
• Clean typography
```

## Animations & Interactions

### Chip Hover Effect
```
Normal State:
╭────────────╮
│ Healthcare │  ← background: light, border: medium
╰────────────╯

Hover:
╭────────────╮
│ Healthcare │  ← lift up 2px, add shadow
╰────────────╯

Selected:
╭────────────╮
│ Healthcare │  ← background: blue-9, color: white
╰────────────╯      border: blue-9, bold
```

### Option Button Animation
```
Unselected:
┌──────────────────────────────┐
│   Clinical-grade accuracy    │  ← light background
└──────────────────────────────┘    gray border

Hover (unselected):
┌──────────────────────────────┐
│   Clinical-grade accuracy    │  ← lift 2px, show shadow
└──────────────────────────────┘    blue border hint

Selected:
┌──────────────────────────────┐
│   Clinical-grade accuracy   │  ← blue background
└──────────────────────────────┘    white text, check icon

Click:
[compress down → spring back up]
```

### Progress Dots
```
Current question:        ●●●●○
                         ─────
                         32px wide, 5px tall
                         smooth transitions
                         
Animation on advance:
●●●○○  →  ●●●●○
   │         │
   └─────────┘
   0.3s cubic-bezier ease
```

### Input Focus
```
Unfocused:
┌─────────────────────────────┐
│ Enter text...               │  ← gray border
└─────────────────────────────┘

Focused:
┌─────────────────────────────┐
│ Enter text...█              │  ← blue border + glow
└─────────────────────────────┘    box-shadow: 0 0 0 3px blue-alpha
```

## Color Palette (iOS-inspired)

```
Primary Colors:
• Blue-9:  #0090FF  (selected states, primary actions)
• Blue-7:  #0EA5E9  (hover hints)
• Blue-3:  #E0F2FE  (chip backgrounds)

Neutral Colors:
• Gray-12: #111111  (primary text)
• Gray-11: #333333  (secondary text)
• Gray-10: #666666  (muted text)
• Gray-6:  #CCCCCC  (borders)
• Gray-5:  #E5E5E5  (disabled states)
• Gray-2:  #F9F9F9  (backgrounds)

Accent Colors:
• Amber-9: #FFB224  (live plan card)
• Amber-6: #FCD34D  (amber border)
• Amber-2: #FFFBEB  (amber background)
• Green-9: #10B981  (success states)
• Violet-9: #8B5CF6 (domain chips)

Transparency:
• Blue-a4: rgba(0, 144, 255, 0.1)  (focus glow)
• Blue-a3: rgba(0, 144, 255, 0.08) (hover tint)
```

## Typography

```
Headings:
• H1: 1.15rem, weight: 600, color: gray-12
• H2: 1.1rem, weight: 600, color: gray-12
• H3: 0.95rem, weight: 600, color: gray-12

Body:
• Large: 0.95rem, weight: 400, color: gray-11
• Normal: 0.88rem, weight: 400, color: gray-11
• Small: 0.82rem, weight: 400, color: gray-10
• Tiny: 0.75rem, weight: 400, color: gray-10

Interactive:
• Button: 0.95rem, weight: 600
• Chip: 0.88rem, weight: 500
• Input: 0.9rem, weight: 400
```

## Spacing & Layout

```
Card Padding: 20px
Card Radius: 16px
Card Shadow: 0 2px 8px rgba(0,0,0,0.05)

Chip Padding: 10px 18px
Chip Radius: 20px

Button Padding: 12px 24px (primary)
Button Padding: 12px 20px (secondary)
Button Radius: 10-12px

Input Padding: 12px 16px
Input Radius: 10px
Input Border: 2px solid

Spacing Scale:
• xs: 4px
• sm: 8px
• md: 12px
• lg: 16px
• xl: 24px
```

## Comparison Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Questions** | Static (5 always same) | Dynamic (AI generated) |
| **Personalization** | None | Based on project context |
| **Feedback** | End only | Real-time plan updates |
| **UI Style** | Basic | iOS-inspired modern |
| **Animations** | None | Smooth transitions |
| **Input Fields** | None | Project name + description |
| **Chip Design** | Simple badges | Interactive cards |
| **Option Buttons** | Plain outlines | Cards with icons |
| **Progress** | Dots only | Dots + live plan |
| **Colors** | Basic theme | Full palette |
| **Error Handling** | Poor | Multiple fallbacks |
