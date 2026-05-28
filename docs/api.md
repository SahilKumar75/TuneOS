# TuneOS API Documentation

This document outlines the API endpoints provided by the TuneOS backend. The backend is built to support the Reflex UI and manage background training tasks.

## Training Endpoints

The primary interactions involve submitting and managing fine-tuning tasks.

### `POST /api/train`
Starts a new fine-tuning job.

**Request Body:**
```json
{
  "model": "string (e.g., 'meta-llama/Llama-2-7b')",
  "dataset": "string (path or huggingface dataset id)",
  "epochs": "integer",
  "batch_size": "integer",
  "learning_rate": "float",
  "lora_config": {
    "r": "integer",
    "lora_alpha": "integer",
    "lora_dropout": "float"
  }
}
```

**Response:**
```json
{
  "job_id": "string",
  "status": "queued"
}
```

### `GET /api/train/{job_id}/status`
Retrieves the status of a specific training job.

**Response:**
```json
{
  "job_id": "string",
  "status": "string (e.g., 'queued', 'running', 'completed', 'failed')",
  "progress": "float (0.0 to 1.0)",
  "logs": "string"
}
```

## Internal Architecture

The TuneOS API relies on Celery and Redis to handle asynchronous training tasks. When a request is made to `/api/train`, a new task is pushed to the Redis queue, which is then picked up by one of the background workers (`trainer/workers`).

*Note: This API is intended for internal use by the TuneOS UI. Stability of the endpoints is not guaranteed for external integrations yet.*
