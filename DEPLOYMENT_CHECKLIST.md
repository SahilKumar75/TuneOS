# Deployment Checklist

Use this checklist before deploying the intent flow improvements.

## Pre-Deployment

### Code Review

- [ ] All files compile without errors
  ```bash
  python -m py_compile app/state/finetune_state.py
  python -m py_compile app/components/finetune/step2_intent.py
  python -m py_compile app/api/datasets_routes.py
  ```

- [ ] No syntax errors in modified files
  ```bash
  reflex run --check
  ```

- [ ] Import statements are correct
- [ ] No unused imports
- [ ] Proper error handling in all async methods

### Environment Setup

- [ ] `.env` file exists
- [ ] `OPENROUTER_API_KEY` documented (even if optional)
  ```bash
  # Add to .env.example
  echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here-optional" >> .env.example
  ```

- [ ] Dependencies are installed
  ```bash
  pip install httpx  # If not already included
  ```

- [ ] No new dependencies that aren't documented

### Documentation

- [ ] All documentation files created:
  - [ ] IMPROVEMENTS_SUMMARY.md
  - [ ] ARCHITECTURE.md
  - [ ] API_REFERENCE.md
  - [ ] TESTING_GUIDE.md
  - [ ] QUICK_REFERENCE.md
  - [ ] UI_EXAMPLES.md
  - [ ] README_CHANGES.md
  - [ ] DEPLOYMENT_CHECKLIST.md (this file)

- [ ] Links between documents work
- [ ] Code examples are accurate
- [ ] Screenshots or ASCII art match current UI

### Backward Compatibility

- [ ] Old state fields still present (for migration)
- [ ] `user_intent` field still populated
- [ ] Existing tests don't break
- [ ] API endpoints maintain same response format

## Testing

### Unit Testing

- [ ] Test with OpenRouter API key
  - [ ] Questions generate successfully
  - [ ] Plan updates after answers
  - [ ] Data generation uses OpenRouter

- [ ] Test without OpenRouter API key
  - [ ] Falls back to default questions
  - [ ] Flow completes without errors
  - [ ] Data generation uses templates

- [ ] Test with invalid API key
  - [ ] Graceful error handling
  - [ ] No crashes
  - [ ] Fallback works

### Integration Testing

- [ ] Complete flow (Phase A  B  C)
  - [ ] Phase A: All inputs work
  - [ ] Transition: Loading state shows
  - [ ] Phase B: Questions display
  - [ ] Phase B: Answers update plan
  - [ ] Phase C: Review shows all data
  - [ ] Approve: Proceeds to next step

- [ ] Different domains tested
  - [ ] Healthcare
  - [ ] Finance
  - [ ] Education
  - [ ] Technology
  - [ ] Legal

- [ ] Edge cases
  - [ ] Empty project name
  - [ ] Very long description
  - [ ] All "Other" custom answers
  - [ ] Rapid clicking
  - [ ] Browser back/forward

### UI/UX Testing

- [ ] Desktop browsers
  - [ ] Chrome
  - [ ] Firefox
  - [ ] Safari
  - [ ] Edge

- [ ] Mobile/tablet (if supported)
  - [ ] Responsive design works
  - [ ] Touch interactions smooth

- [ ] Animations
  - [ ] Chips hover effect
  - [ ] Option buttons animate
  - [ ] Progress dots transition
  - [ ] "Other" expands smoothly

- [ ] Accessibility
  - [ ] Keyboard navigation works
  - [ ] Tab order is logical
  - [ ] Focus indicators visible
  - [ ] Screen reader compatible (basic)

### Performance Testing

- [ ] Question generation < 10 seconds
- [ ] Plan updates < 5 seconds
- [ ] No memory leaks (refresh test)
- [ ] No console errors
- [ ] Network requests complete
- [ ] Animations smooth (60fps)

### Error Handling Testing

- [ ] Network offline
  - [ ] Graceful error messages
  - [ ] Fallbacks work

- [ ] API rate limit hit
  - [ ] Retry logic works
  - [ ] Or falls back

- [ ] Malformed API responses
  - [ ] Doesn't crash
  - [ ] Logs error
  - [ ] Uses fallback

- [ ] Timeout scenarios
  - [ ] Doesn't hang forever
  - [ ] Falls back after timeout

## Deployment

### Pre-Deploy Steps

- [ ] Merge to main branch
  ```bash
  git checkout main
  git merge feature/intent-flow-improvements
  ```

- [ ] Tag release
  ```bash
  git tag -a v1.1.0 -m "Intent flow improvements with AI"
  git push origin v1.1.0
  ```

- [ ] Update CHANGELOG.md
  ```markdown
  ## [1.1.0] - 2024-XX-XX
  ### Added
  - AI-generated personalized questions
  - Live plan updates during question flow
  - iOS-style UI redesign
  - OpenRouter API integration
  - Improved synthetic data generation
  
  ### Changed
  - Phase A now includes project name/description
  - Questions dynamically generated based on context
  - Enhanced error handling with fallbacks
  
  ### Fixed
  - Data generation reliability
  ```

- [ ] Build production
  ```bash
  reflex export --frontend-only
  # or your build command
  ```

### Deploy Steps

- [ ] Deploy to staging first
- [ ] Smoke test on staging
  - [ ] Can access fine-tune wizard
  - [ ] Can complete full flow
  - [ ] No console errors

- [ ] Check staging logs
  - [ ] No error spikes
  - [ ] API calls succeed
  - [ ] Fallbacks working if needed

- [ ] Deploy to production
- [ ] Monitor for 24 hours

### Post-Deploy Verification

- [ ] Production smoke test
  - [ ] Load fine-tune wizard
  - [ ] Complete Phase A
  - [ ] Questions generate
  - [ ] Can answer questions
  - [ ] Review phase works
  - [ ] Can approve and continue

- [ ] Monitor metrics
  - [ ] Error rate (should be low)
  - [ ] API call success rate
  - [ ] Page load times
  - [ ] User completion rate

- [ ] Check logs
  - [ ] No unexpected errors
  - [ ] API calls logging correctly
  - [ ] Fallbacks trigger appropriately

### User Communication

- [ ] Release notes published
- [ ] Users notified of new features
- [ ] Documentation linked in announcement
- [ ] Known issues documented
- [ ] Support team briefed

## Rollback Plan

### If Issues Occur

```bash
# 1. Revert to previous version
git revert <commit-hash>
git push origin main

# 2. Redeploy previous version
reflex export --frontend-only

# 3. Notify users of rollback

# 4. Investigate issues offline
```

### Rollback Triggers

Roll back if:
- [ ] Error rate > 5%
- [ ] Critical bugs found
- [ ] Performance degradation > 50%
- [ ] User complaints spike
- [ ] API costs unexpectedly high

## Monitoring

### What to Monitor

- [ ] API call volumes
  - OpenRouter: /chat/completions
  - Count: questions + plans + data gen

- [ ] Error rates
  - Python exceptions
  - JavaScript console errors
  - API failures

- [ ] Performance
  - Page load times
  - API response times
  - Animation frame rates

- [ ] User behavior
  - Completion rates
  - Drop-off points
  - Feature usage

### Alerts to Set Up

- [ ] Error rate > 5% (alert immediately)
- [ ] API failure rate > 10% (alert)
- [ ] Response time > 30s (alert)
- [ ] OpenRouter rate limit hit (notify)

### Logging

- [ ] Question generation attempts
- [ ] Plan update attempts
- [ ] Fallback usage rates
- [ ] User completion metrics

## Security

### Security Checklist

- [ ] API keys never exposed to frontend
- [ ] API keys not logged
- [ ] User input sanitized (basic)
- [ ] No XSS vulnerabilities
- [ ] HTTPS enforced
- [ ] Rate limiting on endpoints
- [ ] No sensitive data in error messages

### Data Privacy

- [ ] User project descriptions not logged
- [ ] API calls don't include PII
- [ ] OpenRouter privacy policy reviewed
- [ ] Data retention policy documented

## Cost Management

### Cost Monitoring

- [ ] OpenRouter usage dashboard checked
- [ ] Free tier limits understood
  - Questions: ~700 tokens
  - Plans: ~200 tokens
  - Data: ~2000 tokens
  - Total: ~2,900 per user per session

- [ ] Cost alerts set (if using paid tier)
- [ ] Fallback methods ready (always free)

### Optimization

- [ ] Unnecessary API calls removed
- [ ] Caching considered (future)
- [ ] Batching where possible (future)
- [ ] Timeout values reasonable

## Documentation

### Internal Docs

- [ ] Team trained on new flow
- [ ] Support docs updated
- [ ] Architecture diagrams shared
- [ ] API docs accessible

### External Docs

- [ ] User guide updated
- [ ] API reference published
- [ ] FAQ section created
- [ ] Video tutorial (optional)

## Success Metrics

### Define Success

Track these for 30 days:

- [ ] User completion rate (target: maintain or improve)
- [ ] Average session time (target: similar or less)
- [ ] User satisfaction (survey if possible)
- [ ] Error rate (target: < 2%)
- [ ] API success rate (target: > 95%)
- [ ] Feature adoption (% using new features)

### Review Schedule

- [ ] Day 1: Immediate issues check
- [ ] Day 7: First week review
- [ ] Day 14: Mid-month check
- [ ] Day 30: Full feature review

## Final Checks

### Before Going Live

- [ ] All tests pass
- [ ] Documentation complete
- [ ] Team trained
- [ ] Monitoring set up
- [ ] Rollback plan ready
- [ ] API keys secured
- [ ] Backup created
- [ ] Announcement ready

### Go/No-Go Decision

**GO if:**
- All critical tests pass
- No blocking bugs
- Performance acceptable
- Rollback plan ready
- Team ready

**NO-GO if:**
- Critical bugs exist
- Performance issues
- Security concerns
- Team not ready
- Documentation incomplete

---

## Deployment Sign-Off

- [ ] **Developer:** Code reviewed and tested
- [ ] **QA:** All tests pass
- [ ] **Design:** UI/UX approved
- [ ] **Product:** Features meet requirements
- [ ] **Security:** No security concerns
- [ ] **DevOps:** Infrastructure ready

**Deployment Date:** _______________

**Deployed By:** _______________

**Approved By:** _______________

---

## Post-Deployment

### Week 1 Checklist

- [ ] Monitor error rates daily
- [ ] Review user feedback
- [ ] Check API usage
- [ ] Fix any quick wins
- [ ] Update docs based on questions

### Week 2-4 Checklist

- [ ] Analyze completion rates
- [ ] Review performance metrics
- [ ] Plan improvements
- [ ] Consider user feedback
- [ ] Optimize costs if needed

---

**Good luck with the deployment! **
