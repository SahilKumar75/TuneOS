# TuneOS Intent Flow Improvements

## Summary of Changes

I've completely overhauled the user intent collection flow with these major improvements:

### 1. **Dynamic, Personalized Questions with OpenRouter API** ✨

**Problem:** Questions were static and not relevant to user's specific project.

**Solution:**
- Added `_generate_personalized_questions()` method that uses OpenRouter API
- Questions are now generated based on Phase A inputs (project name, description, domain, etc.)
- Uses `deepseek/deepseek-v4-flash:free` model for intelligent question generation
- Fallback to default questions if API fails or key is missing

**Key Changes:**
- `finetune_state.py`: New state fields
  - `intent_questions: list[dict]` - Dynamically generated questions
  - `intent_is_generating_questions: bool` - Loading state
  - `intent_live_plan: str` - Updates as user answers
- Questions adapt to healthcare, finance, education, tech domains automatically

### 2. **Live Plan Updates** 🔄

**Problem:** User didn't see how their answers affected the final plan until the end.

**Solution:**
- Added `_update_live_plan()` method that runs after each answer
- Uses OpenRouter API to generate a concise 2-3 sentence summary
- Displays in an iOS-style amber card above the current question
- Updates in real-time as user progresses through questions

**Benefits:**
- User sees immediate feedback
- Can adjust answers if plan doesn't match expectations
- More engaging and interactive experience

### 3. **iOS-Style UI Overhaul** 🎨

**Problem:** UI felt basic and not modern.

**Solution:** Complete redesign with iOS-inspired components:

#### Phase A (Context Collection):
- Circular numbered badge (step indicator)
- Improved filter chips with smooth animations
  - Hover effects (lift on hover)
  - Active state animations
  - Better color schemes and borders
- Enhanced input fields with focus states
- Project name and description text inputs
- Better spacing and visual hierarchy

#### Phase B (Questions):
- iOS-style option buttons with check icons
- Smooth slide-down animation for "Other" input
- Progress dots with smooth transitions
- Card-based layout with rounded corners and shadows
- Better button styling with hover/active states
- Live plan preview in amber notification card

#### Phase C (Review):
- Enhanced markdown preview
- Better navigation buttons

### 4. **Improved Synthetic Data Generation** 🤖

**Problem:** Data generation had poor error handling and limited methods.

**Solution:**
- Added OpenRouter integration as primary method
- Created `_openrouter_generate()` function
- Implements fallback hierarchy:
  1. OpenRouter (preferred - faster, more reliable)
  2. HuggingFace Inference API
  3. Template-based generation (guaranteed fallback)
- Better error logging and debugging
- Returns generation method and errors in stats

**Key Improvements:**
- `datasets_routes.py`: Async/await for OpenRouter calls
- Error tracking: `error_log` array in response
- JSON extraction with markdown removal
- Sample validation before returning
- Higher token limits for better quality

### 5. **State Management Improvements** 📊

**New State Fields:**
```python
intent_questions: list[dict] = []  # Dynamic questions
intent_is_generating_questions: bool = False
intent_live_plan: str = ""  # Real-time plan updates
```

**Updated Methods:**
- `intent_next_phase()` - Now async, generates questions on Phase A → B transition
- `set_intent_answer()` - Now async, triggers live plan update
- `_generate_intent_md()` - Uses live plan and dynamic questions

## How It Works Now

### User Flow:

1. **Phase A - Project Context** (iOS-style form)
   - Enter project name and description
   - Select use case, domain, task type with animated chips
   - Click "Continue to Questions"

2. **Transition** (Loading state)
   - OpenRouter API generates 5 personalized questions
   - Based on project details from Phase A
   - Shows spinner with "Generating personalized questions..." message

3. **Phase B - Dynamic Questions** (One at a time)
   - Each question has 3-4 relevant options
   - Options styled as iOS cards with check icons
   - "Other" option expands smoothly for custom input
   - **Live Plan** appears in amber card above question after first answer
   - Plan updates after each answer
   - Progress dots show position (1 of 5, 2 of 5, etc.)
   - Smooth scrolling between questions

4. **Phase C - Review & Approve**
   - Shows complete intent profile with live plan as summary
   - Includes all context and answers
   - Edit or approve to continue

### Technical Flow:

```
Phase A Submit
    ↓
_generate_personalized_questions()
    ↓
OpenRouter API Call
    ↓
[{heading: "...", options: ["...", "..."]}]
    ↓
Phase B Display
    ↓
User Selects Answer
    ↓
_update_live_plan()
    ↓
OpenRouter API Call
    ↓
Updated Plan Summary
    ↓
Display in Amber Card
```

## API Integration Details

### OpenRouter Endpoints Used:

1. **Question Generation:**
   - Model: `deepseek/deepseek-v4-flash:free`
   - Temperature: 0.7
   - Max tokens: 1500
   - Returns: JSON array of question objects

2. **Plan Updates:**
   - Model: `deepseek/deepseek-v4-flash:free`
   - Temperature: 0.5
   - Max tokens: 200
   - Returns: Plain text summary

3. **Synthetic Data:**
   - Model: `deepseek/deepseek-v4-flash:free`
   - Temperature: 0.8
   - Max tokens: n * 150 (scaled to dataset size)
   - Returns: JSON array of instruction/output pairs

## Configuration Required

Add to your `.env` file:
```bash
OPENROUTER_API_KEY=your_key_here
```

The system gracefully falls back to defaults if the key is missing.

## Files Modified

1. **`app/state/finetune_state.py`**
   - Added dynamic question state fields
   - Made methods async for API calls
   - Added `_generate_personalized_questions()`
   - Added `_update_live_plan()`
   - Updated `_generate_intent_md()` to use live plan

2. **`app/components/finetune/step2_intent.py`**
   - Complete UI redesign with iOS styling
   - Dynamic question rendering with `rx.foreach`
   - Loading states for question generation
   - Live plan preview component
   - Enhanced animations and transitions
   - Removed hardcoded `_QUESTIONS` array

3. **`app/api/datasets_routes.py`**
   - Added `_openrouter_generate()` function
   - Made `generate_dataset()` async
   - Implemented fallback hierarchy
   - Added error logging
   - Better JSON extraction and validation

## Benefits

✅ **Personalized Experience** - Questions match user's specific project
✅ **Real-time Feedback** - Live plan updates show immediate impact
✅ **Better UX** - iOS-inspired design feels modern and polished
✅ **More Reliable** - Multiple fallbacks for data generation
✅ **Better Debugging** - Error logs help diagnose issues
✅ **Scalable** - Can easily add more AI-powered features
✅ **Adaptive** - Works without OpenRouter key (graceful degradation)

## Testing Recommendations

1. **Test with OpenRouter API key:**
   - Verify question generation works
   - Check plan updates after answers
   - Test synthetic data generation

2. **Test without API key:**
   - Verify fallback to default questions
   - Check template-based data generation still works

3. **Test edge cases:**
   - Very short project descriptions
   - All optional fields left blank
   - Custom "Other" answers
   - API timeout/errors

4. **UI Testing:**
   - Verify animations are smooth
   - Check responsive design
   - Test keyboard navigation
   - Verify accessibility

## Next Steps

Consider these future enhancements:

1. **Question History** - Save and reuse question sets
2. **Multi-language Support** - Generate questions in user's language
3. **Question Refinement** - Let user edit generated questions
4. **A/B Testing** - Track which question types work best
5. **Analytics** - Monitor API usage and success rates
6. **Caching** - Cache generated questions for similar intents
7. **Synthetic Data from Intent** - Use full intent profile for better data
