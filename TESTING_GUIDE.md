# Testing Guide for Intent Flow Improvements

## Quick Start

### 1. Setup Environment Variables

Add to your `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get a free API key from: https://openrouter.ai/

### 2. Start the Application

```bash
# If using reflex
reflex run

# Or if using another command
python main.py
```

### 3. Navigate to Fine-Tune Wizard

Go to the Fine-tune section and start a new project.

## Test Scenarios

###  Test 1: Full Flow with OpenRouter

**Goal:** Verify personalized questions are generated

**Steps:**
1. Fill out Phase A:
   - Project Name: "Medical Q&A Bot"
   - Description: "Answer patient questions about diabetes management"
   - Use Case: Personal
   - Domain: Healthcare
   - Task Type: Text generation

2. Click "Continue to Questions"

3. **Expected:**
   - Loading spinner appears: "Generating personalized questions..."
   - After 2-5 seconds, question 1 appears
   - Questions should be healthcare/diabetes specific
   - Example: "What level of medical accuracy is required?"

4. Answer first question

5. **Expected:**
   - Live Plan amber card appears above question 2
   - Plan summary mentions diabetes, healthcare, Q&A
   - Example: "A text generation model for healthcare that provides accurate diabetes management information to patients."

6. Answer remaining questions

7. **Expected:**
   - Plan updates after each answer
   - Progress dots advance (1/5, 2/5, etc.)
   - Smooth scroll to next question

8. Review Phase C

9. **Expected:**
   - Summary includes live plan
   - All answers displayed
   - Project details shown

###  Test 2: Fallback (No API Key)

**Goal:** Verify graceful degradation

**Steps:**
1. Remove or comment out `OPENROUTER_API_KEY` in `.env`
2. Restart application
3. Go through Phase A
4. Click "Continue to Questions"

**Expected:**
- Default questions appear immediately (no loading)
- Questions are generic:
  - "What is the primary goal of this model?"
  - "Who is the target audience?"
  - etc.
- No live plan updates (amber card doesn't appear)
- Flow still works end-to-end

###  Test 3: Different Domains

**Goal:** Verify questions adapt to different domains

**Test each domain:**

**Finance:**
- Project: "Investment Advisor"
- Domain: Finance
- Expected questions about: risk tolerance, regulatory compliance, financial accuracy

**Legal:**
- Project: "Contract Analyzer"
- Domain: Legal
- Expected questions about: jurisdiction, legal accuracy, citation requirements

**Education:**
- Project: "Homework Helper"
- Domain: Education
- Expected questions about: grade level, learning objectives, pedagogy

**Code:**
- Project: "Code Review Bot"
- Task Type: Code
- Expected questions about: programming languages, code style, review depth

###  Test 4: UI/UX Testing

**Goal:** Verify iOS-style design works

**Phase A:**
- [ ] Chips animate on hover (lift up)
- [ ] Chips change color when selected
- [ ] Text inputs have focus states (blue border + shadow)
- [ ] Continue button has hover effect
- [ ] Circular "1" badge displays

**Phase B:**
- [ ] Progress dots animate smoothly
- [ ] Option buttons show check icon when selected
- [ ] "Other" expands smoothly with animation
- [ ] Custom input has focus state
- [ ] Back/Continue buttons styled correctly
- [ ] Live plan amber card displays properly
- [ ] Questions scroll smoothly

**Phase C:**
- [ ] Markdown renders correctly
- [ ] Edit button works
- [ ] Approve button proceeds to next step

###  Test 5: Edge Cases

**Empty Fields:**
- Leave all Phase A fields blank
- Click Continue
- Expected: Generic questions generated

**Very Long Description:**
- Enter 500+ character description
- Expected: Questions still relevant, no errors

**Custom "Other" Answers:**
- Select "Other" for every question
- Enter custom text
- Expected: Plan updates with custom answers

**API Timeout:**
- Use slow network connection
- Expected: Falls back to default questions after timeout

**Rapid Clicking:**
- Click through questions very fast
- Expected: No duplicate plan updates, smooth navigation

###  Test 6: Synthetic Data Generation

**Goal:** Verify data generation uses OpenRouter

**Steps:**
1. Complete intent flow with meaningful answers
2. Go to data generation step
3. Generate 10 samples

**Expected:**
- Generation completes in <30 seconds
- Samples are high quality and relevant
- Stats show `"generation_method": "openrouter"`
- No errors in console

**Without API Key:**
- Stats show `"generation_method": "template"`
- Still generates samples (lower quality)

## Debugging Tips

### Check Console Logs

Look for these messages:
```python
print(f"Error generating questions: {e}")  # Question generation failed
print(f"Error updating live plan: {e}")    # Plan update failed
```

### Check Network Tab

OpenRouter API calls should show:
- POST to `https://openrouter.ai/api/v1/chat/completions`
- Status 200
- Response with JSON content

### Common Issues

**Issue:** Questions not generating
- **Check:** `OPENROUTER_API_KEY` in .env
- **Check:** Network connectivity
- **Check:** API key is valid (not expired)
- **Solution:** System falls back to defaults automatically

**Issue:** Plan not updating
- **Check:** Console for error messages
- **Check:** Network tab for API calls
- **Solution:** Feature is optional, flow continues without it

**Issue:** UI looks broken
- **Check:** CSS variables are defined (--blue-9, --gray-5, etc.)
- **Check:** Radix UI theme is loaded
- **Solution:** Check browser console for CSS errors

**Issue:** Synthetic data fails
- **Check:** Both OPENROUTER_API_KEY and HF_TOKEN if available
- **Solution:** Will fall back to template method automatically

## Performance Benchmarks

**Question Generation:**
- With OpenRouter: 2-5 seconds
- Fallback: Instant

**Plan Updates:**
- Per answer: 1-2 seconds
- Async, doesn't block UI

**Synthetic Data:**
- OpenRouter (10 samples): 10-20 seconds
- HuggingFace (10 samples): 15-30 seconds
- Template (10 samples): <1 second

## Success Criteria

All tests should pass with:
-  No Python errors
-  No JavaScript console errors
-  Smooth animations (60fps)
-  Questions relevant to context
-  Plan updates accurately
-  Graceful fallbacks work
-  UI matches iOS design patterns

## Report Issues

When reporting issues, include:
1. Test scenario number
2. Expected vs actual behavior
3. Console logs (Python and Browser)
4. Network requests (if API related)
5. Environment details (OS, browser, Python version)
