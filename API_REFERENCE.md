# API Reference - OpenRouter Integration

## Overview

TuneOS now integrates with OpenRouter API for three key features:
1. **Personalized Question Generation** (Phase A → B)
2. **Live Plan Updates** (Phase B, per answer)
3. **Synthetic Data Generation** (Data step)

All use the `deepseek/deepseek-v4-flash:free` model (free tier).

---

## 1. Question Generation API

### Purpose
Generate 5 personalized questions based on user's project context.

### When Called
Automatically when user clicks "Continue to Questions" in Phase A.

### Request

```http
POST https://openrouter.ai/api/v1/chat/completions
Content-Type: application/json
Authorization: Bearer YOUR_OPENROUTER_KEY
X-Title: TuneOS Intent Questions
```

```json
{
  "model": "deepseek/deepseek-v4-flash:free",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI that generates JSON only. Never include explanations, just pure JSON."
    },
    {
      "role": "user",
      "content": "Generate 5 personalized questions for a fine-tuning project with these details:\n- Project Name: Medical Q&A Bot\n- Description: Answer patient questions about diabetes\n- Use Case: personal\n- Domain: healthcare\n- Task Type: text\n- Expected Volume: Not specified\n- Accuracy Requirements: Standard\n\nGenerate 5 highly relevant questions that will help refine this fine-tuning project. Each question should have 3-4 specific answer options.\n\nReturn ONLY valid JSON in this exact format, no other text:\n{\n  \"questions\": [\n    {\n      \"heading\": \"Your question here?\",\n      \"options\": [\"Option 1\", \"Option 2\", \"Option 3\"]\n    }\n  ]\n}"
    }
  ],
  "max_tokens": 1500,
  "temperature": 0.7
}
```

### Response

```json
{
  "id": "gen-abc123",
  "model": "deepseek/deepseek-v4-flash:free",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "{\n  \"questions\": [\n    {\n      \"heading\": \"What level of medical accuracy is required?\",\n      \"options\": [\n        \"General health information\",\n        \"Clinical-grade accuracy\",\n        \"Patient education level\"\n      ]\n    },\n    {\n      \"heading\": \"Who is the primary target audience?\",\n      \"options\": [\n        \"Patients with diabetes\",\n        \"Healthcare professionals\",\n        \"Caregivers and family members\"\n      ]\n    },\n    {\n      \"heading\": \"What type of diabetes questions should it handle?\",\n      \"options\": [\n        \"Type 1 diabetes specific\",\n        \"Type 2 diabetes specific\",\n        \"Both types and general diabetes info\"\n      ]\n    },\n    {\n      \"heading\": \"What tone should the responses have?\",\n      \"options\": [\n        \"Empathetic and supportive\",\n        \"Formal and medical\",\n        \"Simple and educational\"\n      ]\n    },\n    {\n      \"heading\": \"How will you measure success?\",\n      \"options\": [\n        \"Patient satisfaction scores\",\n        \"Accuracy of medical information\",\n        \"Response clarity and usefulness\"\n      ]\n    }\n  ]\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 245,
    "completion_tokens": 380,
    "total_tokens": 625
  }
}
```

### Error Handling

```python
try:
    # Make API call
    resp = await http.post(...)
    
    if resp.status_code == 200:
        # Extract JSON from response
        content = data["choices"][0]["message"]["content"]
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            parsed = json.loads(json_match.group())
            questions = parsed.get("questions", [])[:5]
        else:
            raise ValueError("No JSON found")
    else:
        raise Exception(f"API returned {resp.status_code}")
        
except Exception as e:
    print(f"Error generating questions: {e}")
    # Fallback to default questions
    questions = DEFAULT_QUESTIONS
```

### Cost
- **Free tier**: deepseek/deepseek-v4-flash:free
- **Tokens**: ~600-800 per request
- **Rate limit**: Check OpenRouter docs

---

## 2. Live Plan Update API

### Purpose
Generate a 2-3 sentence summary of the user's plan based on answers so far.

### When Called
Automatically after user selects an answer in Phase B.

### Request

```http
POST https://openrouter.ai/api/v1/chat/completions
Content-Type: application/json
Authorization: Bearer YOUR_OPENROUTER_KEY
X-Title: TuneOS Plan Update
```

```json
{
  "model": "deepseek/deepseek-v4-flash:free",
  "messages": [
    {
      "role": "system",
      "content": "You write concise, clear summaries. Never use markdown or formatting."
    },
    {
      "role": "user",
      "content": "Based on these project details and answers, write a concise 2-3 sentence summary of what this fine-tuned model will do:\n\nProject Context:\n- Name: Medical Q&A Bot\n- Description: Answer patient questions about diabetes\n- Domain: healthcare\n- Task: text\n\nAnswered Questions:\nQ: What level of medical accuracy is required?\nA: Clinical-grade accuracy\n\nQ: Who is the primary target audience?\nA: Healthcare professionals\n\nWrite ONLY the summary, no other text."
    }
  ],
  "max_tokens": 200,
  "temperature": 0.5
}
```

### Response

```json
{
  "choices": [
    {
      "message": {
        "content": "A healthcare text generation model that provides clinical-grade diabetes management information to healthcare professionals. The model will answer questions with medical accuracy suitable for clinical settings, helping professionals make informed decisions about patient care."
      }
    }
  ]
}
```

### Usage Pattern

```python
# User answers Question 1
set_intent_answer(0, "Clinical-grade accuracy")
    ↓
_update_live_plan()  # Async call
    ↓
Plan: "A model that provides clinical-grade information..."

# User answers Question 2
set_intent_answer(1, "Healthcare professionals")
    ↓
_update_live_plan()  # Async call with updated context
    ↓
Plan: "A healthcare model for professionals that provides
      clinical-grade information..."  ← More refined

# User answers Question 3
set_intent_answer(2, "Both types of diabetes")
    ↓
_update_live_plan()
    ↓
Plan: "A comprehensive healthcare model for professionals
      covering all types of diabetes with clinical-grade
      accuracy..."  ← Most complete
```

### Error Handling

```python
try:
    resp = await http.post(...)
    if resp.status_code == 200:
        summary = data["choices"][0]["message"]["content"].strip()
        self.intent_live_plan = summary
except Exception as e:
    print(f"Error updating live plan: {e}")
    # Silent fail - plan update is optional
    # User flow continues without disruption
```

### Cost
- **Free tier**: deepseek/deepseek-v4-flash:free
- **Tokens**: ~150-300 per request
- **Frequency**: Up to 5 times (once per question)

---

## 3. Synthetic Data Generation API

### Purpose
Generate training data samples (instruction/output pairs).

### When Called
When user clicks "Generate Data" in the data generation step.

### Request

```http
POST https://openrouter.ai/api/v1/chat/completions
Content-Type: application/json
Authorization: Bearer YOUR_OPENROUTER_KEY
X-Title: TuneOS Dataset Generation
```

```json
{
  "model": "deepseek/deepseek-v4-flash:free",
  "messages": [
    {
      "role": "system",
      "content": "You generate JSON datasets only. Never use markdown formatting."
    },
    {
      "role": "user",
      "content": "You are a dataset creator. Generate 10 diverse, high-quality instruction/output pairs for fine-tuning a language model.\n\nUser Intent: A healthcare model for diabetes Q&A\n\nExample pairs:\n- Instruction: What are symptoms of Type 2 diabetes?\n  Output: Common symptoms include frequent urination...\n- Instruction: How often should diabetics check blood sugar?\n  Output: Most people with Type 2 diabetes should check...\n\nGenerate 10 NEW examples (not repeating the seeds) in this JSON format:\n[{\"instruction\": \"...\", \"output\": \"...\"}]\n\nMake the instructions diverse, covering different aspects of the intent.\nReturn ONLY the JSON array, no markdown, no explanations."
    }
  ],
  "max_tokens": 2000,
  "temperature": 0.8
}
```

### Response

```json
{
  "choices": [
    {
      "message": {
        "content": "[\n  {\n    \"instruction\": \"What is the difference between Type 1 and Type 2 diabetes?\",\n    \"output\": \"Type 1 diabetes is an autoimmune condition where the pancreas produces little to no insulin, typically diagnosed in children and young adults. Type 2 diabetes develops when the body becomes resistant to insulin or doesn't produce enough, usually appearing in adults and often linked to lifestyle factors.\"\n  },\n  {\n    \"instruction\": \"Can diabetes be reversed?\",\n    \"output\": \"Type 2 diabetes can sometimes be put into remission through significant lifestyle changes including weight loss, diet modification, and increased physical activity. Type 1 diabetes cannot be reversed as it's an autoimmune condition, but it can be effectively managed with insulin therapy.\"\n  }\n  // ... 8 more samples\n]"
      }
    }
  ]
}
```

### Processing Pipeline

```python
async def _openrouter_generate(intent, n, seeds, api_key):
    # 1. Make API call
    resp = await http.post(...)
    
    # 2. Extract content
    content = resp.json()["choices"][0]["message"]["content"]
    
    # 3. Remove markdown (if present)
    content = re.sub(r'```json\s*|\s*```', '', content)
    
    # 4. Extract JSON array
    json_match = re.search(r'\[.*?\]', content, re.DOTALL)
    samples = json.loads(json_match.group())
    
    # 5. Validate structure
    valid_samples = []
    for s in samples:
        if "instruction" in s and "output" in s:
            valid_samples.append(s)
    
    return valid_samples
```

### Fallback Chain

```
1. Try OpenRouter (preferred)
   ├── Success → Return samples
   └── Fail → Log error, try next
   
2. Try HuggingFace (if HF_TOKEN set)
   ├── Success → Return samples
   └── Fail → Log error, try next
   
3. Use Templates (guaranteed)
   └── Always succeeds with basic samples
```

### Response Stats

```json
{
  "samples": [...],
  "dataset_path": "storage/datasets/generated_abc123.jsonl",
  "stats": {
    "total_generated": 10,
    "final_count": 10,
    "diversity_score": 0.87,
    "generation_method": "openrouter",
    "errors": null
  }
}
```

### Cost
- **Free tier**: deepseek/deepseek-v4-flash:free
- **Tokens**: ~1500-3000 per request (varies with n)
- **Frequency**: Once per generation (user triggered)

---

## Rate Limits & Best Practices

### OpenRouter Free Tier
```
Model: deepseek/deepseek-v4-flash:free
- Requests per minute: ~20
- Tokens per request: No hard limit
- Daily quota: Check OpenRouter dashboard
```

### Optimization Tips

**1. Batch Operations:**
```python
# Bad: Multiple plan updates in quick succession
for answer in answers:
    await _update_live_plan()  # 5 API calls

# Good: Debounce or batch
await _update_live_plan()  # 1 API call after all answers
```

**2. Caching:**
```python
# Cache generated questions for same context
cache_key = hash(f"{project_name}:{domain}:{task_type}")
if cache_key in question_cache:
    return question_cache[cache_key]
```

**3. Timeout Handling:**
```python
async with httpx.AsyncClient(timeout=30.0) as http:
    # Will timeout after 30s and fallback
```

**4. Graceful Degradation:**
```python
# Always have a fallback
try:
    result = await api_call()
except Exception:
    result = fallback_method()
```

---

## Environment Variables

### Required
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Optional
```bash
HF_TOKEN=hf_your_huggingface_token  # For data generation fallback
```

### Testing
```bash
# Verify API key works
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-v4-flash:free",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
```

---

## Error Codes & Debugging

### Common HTTP Status Codes

**200 - Success**
```json
{
  "choices": [{
    "message": {"content": "..."}
  }]
}
```

**401 - Unauthorized**
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error"
  }
}
```
→ Check `OPENROUTER_API_KEY` in .env

**429 - Rate Limit**
```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error"
  }
}
```
→ Wait and retry, or use fallback

**503 - Service Unavailable**
```json
{
  "error": {
    "message": "Service temporarily unavailable"
  }
}
```
→ Retry or use fallback

### Debugging Checklist

```bash
# 1. Check API key is set
env | grep OPENROUTER_API_KEY

# 2. Test API connectivity
curl https://openrouter.ai/api/v1/models

# 3. Check Python logs
tail -f .web/reflex.log | grep "Error"

# 4. Monitor network requests
# Open browser DevTools → Network tab
# Filter: "openrouter.ai"
# Check request/response

# 5. Verify model availability
curl https://openrouter.ai/api/v1/models | grep deepseek
```

### Console Log Messages

**Success:**
```
No log (silent success)
```

**Failure:**
```python
print(f"Error generating questions: {e}")
# Fallback triggered automatically

print(f"Error updating live plan: {e}")
# Silent fail, flow continues

print(f"OpenRouter failed: {str(e)}")
# Tries next method in fallback chain
```

---

## API Call Summary

| Feature | Endpoint | Model | Tokens | Frequency | Fallback |
|---------|----------|-------|--------|-----------|----------|
| Question Gen | /chat/completions | deepseek-v4 | ~700 | Once per Phase A | Default questions |
| Plan Updates | /chat/completions | deepseek-v4 | ~200 | 5x per session | Silent (optional) |
| Data Gen | /chat/completions | deepseek-v4 | ~2000 | On demand | HF → Templates |

**Total tokens per complete flow:** ~2,700 (free tier)

---

## Testing API Calls

### Test Question Generation
```python
import httpx
import os

async def test_questions():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Test",
            },
            json={
                "model": "deepseek/deepseek-v4-flash:free",
                "messages": [
                    {"role": "system", "content": "Generate JSON only"},
                    {"role": "user", "content": "Generate 2 test questions"}
                ],
                "max_tokens": 500,
            },
        )
        
        print(resp.status_code)
        print(resp.json())

# Run test
import asyncio
asyncio.run(test_questions())
```

### Monitor API Usage
- Dashboard: https://openrouter.ai/activity
- Shows: requests, tokens, costs, errors
- Free tier: $0 cost but has limits

---

# Training Job API

The following endpoints are served by the FastAPI backend (default port 8000).
All job endpoints return `{"job_id": "<uuid>"}` on success and a `4xx` JSON error
body on validation failure.

---

## GET /api/health

Returns the operational status of the API, Redis broker, and worker pool.

### Response

```json
{
  "status": "ok",
  "redis": true,
  "worker_count": 2
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` when the service is healthy |
| `redis` | `bool` | `true` when the Redis broker is reachable |
| `worker_count` | `int` | Number of Celery workers currently registered |

---

## POST /api/jobs

Submit a supervised fine-tuning (SFT) job. The job is placed on the `sft` Celery queue.

### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `string` | yes | Hugging Face model ID (e.g. `mistralai/Mistral-7B-v0.1`) |
| `dataset_path` | `string` | one of | Path to a local `.jsonl` dataset file |
| `hub_dataset_id` | `string` | one of | Hugging Face dataset ID |
| `lora_rank` | `int` | no | LoRA rank `r` (default `8`) |
| `lora_alpha` | `int` | no | LoRA scaling alpha (default `16`) |
| `lora_dropout` | `float` | no | LoRA dropout (default `0.05`) |
| `epochs` | `int` | no | Training epochs (default `3`) |
| `batch_size` | `int` | no | Per-device train batch size (default `4`) |
| `learning_rate` | `float` | no | AdamW learning rate (default `2e-4`) |
| `bf16` | `bool` | no | Enable bfloat16 mixed precision |
| `seed` | `int` | no | Global random seed (default `42`) |
| `eval_split_ratio` | `float` | no | Fraction of data reserved for eval; `0` skips eval |
| `eval_steps` | `int` | no | Run evaluation every N steps |
| `prompt_template` | `string` | no | One of `alpaca`, `chatml`, `llama3`, `phi3`, `zephyr` |
| `packing` | `bool` | no | Enable SFTTrainer sample packing |
| `compute_backend` | `string` | no | `local`, `modal`, or `zerogpu` (default `local`) |
| `experiment_id` | `string` | no | Tag this run under an existing experiment |

### Response

```json
{ "job_id": "b3f2a1c0-..." }
```

---

## POST /api/jobs/dpo

Submit a Direct Preference Optimization job. The job is placed on the `dpo` Celery queue.

### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `string` | yes | Base model Hugging Face ID |
| `dataset_path` | `string` | one of | Path to a local preference dataset file |
| `hub_dataset_id` | `string` | one of | Hugging Face dataset ID |
| `prompt_col` | `string` | no | Column name for the prompt (default `"prompt"`) |
| `chosen_col` | `string` | no | Column name for the chosen response (default `"chosen"`) |
| `rejected_col` | `string` | no | Column name for the rejected response (default `"rejected"`) |
| `beta` | `float` | no | KL-penalty coefficient (default `0.1`) |
| `max_length` | `int` | no | Maximum total sequence length (default `1024`) |
| `max_prompt_length` | `int` | no | Maximum prompt length (default `512`) |
| `lora_rank` | `int` | no | LoRA rank `r` (default `8`) |
| `lora_alpha` | `int` | no | LoRA scaling alpha (default `16`) |
| `lora_dropout` | `float` | no | LoRA dropout (default `0.05`) |
| `epochs` | `int` | no | Training epochs (default `1`) |
| `batch_size` | `int` | no | Per-device train batch size (default `4`) |
| `bf16` | `bool` | no | Enable bfloat16 mixed precision |
| `seed` | `int` | no | Global random seed (default `42`) |
| `experiment_id` | `string` | no | Tag this run under an existing experiment |

### Response

```json
{ "job_id": "c7e4d2f1-..." }
```

---

## POST /api/jobs/distill

Submit a knowledge distillation job. The student model is fine-tuned to match the
teacher model's output distribution. The job is placed on the `kd` Celery queue.

### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `string` | yes | Student model Hugging Face ID |
| `teacher_model` | `string` | yes | Teacher model Hugging Face ID |
| `dataset_path` | `string` | one of | Path to a local `.jsonl` dataset file |
| `hub_dataset_id` | `string` | one of | Hugging Face dataset ID |
| `temperature` | `float` | no | Softmax temperature for distillation (default `2.0`) |
| `alpha` | `float` | no | Weight on the distillation loss vs. CE loss (default `0.5`) |
| `lora_rank` | `int` | no | LoRA rank `r` (default `8`) |
| `lora_alpha` | `int` | no | LoRA scaling alpha (default `16`) |
| `lora_dropout` | `float` | no | LoRA dropout (default `0.05`) |
| `epochs` | `int` | no | Training epochs (default `3`) |
| `batch_size` | `int` | no | Per-device train batch size (default `4`) |
| `seed` | `int` | no | Global random seed (default `42`) |
| `experiment_id` | `string` | no | Tag this run under an existing experiment |

### Response

```json
{ "job_id": "a1b2c3d4-..." }
```

---

## POST /api/jobs/vision

Submit a vision-language model fine-tuning job. The dataset must contain an image
column and text columns; images are preprocessed via `AutoProcessor`.

### Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `string` | yes | Multimodal model Hugging Face ID |
| `dataset_path` | `string` | one of | Path to a local dataset directory or file |
| `hub_dataset_id` | `string` | one of | Hugging Face dataset ID |
| `image_col` | `string` | no | Column name for images (default `"image"`) |
| `instruction_col` | `string` | no | Column name for instruction text (default `"instruction"`) |
| `output_col` | `string` | no | Column name for target output text (default `"output"`) |
| `modality` | `string` | no | Must be `"vision"` (used for queue routing) |
| `use_4bit` | `bool` | no | Enable 4-bit quantization via bitsandbytes |
| `lora_rank` | `int` | no | LoRA rank `r` (default `8`) |
| `epochs` | `int` | no | Training epochs (default `3`) |
| `batch_size` | `int` | no | Per-device train batch size (default `2`) |

### Response

```json
{ "job_id": "f9e8d7c6-..." }
```
