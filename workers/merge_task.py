"""Celery tasks for post-training operations: merge, GGUF export, GitHub push."""

import json
import os
import traceback

import redis

from workers.celery_app import celery_app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import spaces
    _gpu_decorator = spaces.GPU
except ImportError:
    _gpu_decorator = lambda fn: fn  # noqa: E731


def _publish_deploy_log(r: redis.Redis, job_id: str, message: str):
    r.publish(f"job:{job_id}:deploy", json.dumps({"message": message}))


@_gpu_decorator
def _run_merge_impl(job_id: str, base_model_id: str, adapter_path: str, output_path: str, hf_token: str = ""):
    from trainer.merge import merge_adapter
    return merge_adapter(base_model_id, adapter_path, output_path, hf_token)


@celery_app.task(bind=True, name="workers.merge_task.merge_adapter")
def merge_adapter_task(self, job_id: str, base_model_id: str, adapter_path: str, hf_token: str = ""):
    r = redis.from_url(REDIS_URL)
    merged_key = f"job:{job_id}:merged"
    output_path = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), job_id, "merged")

    try:
        _publish_deploy_log(r, job_id, "Starting model merge — this may take several minutes...")
        merged_path = _run_merge_impl(job_id, base_model_id, adapter_path, output_path, hf_token)
        r.set(merged_key, json.dumps({"status": "done", "merged_path": merged_path}))
        _publish_deploy_log(r, job_id, f"Merge complete. Saved to {merged_path}")
        return merged_path
    except Exception as e:
        r.set(merged_key, json.dumps({"status": "failed", "error": str(e)}))
        _publish_deploy_log(r, job_id, f"Merge failed: {e}")
        raise


@celery_app.task(bind=True, name="workers.merge_task.export_gguf")
def export_gguf_task(self, job_id: str, merged_model_path: str, quant_type: str = "Q4_K_M"):
    r = redis.from_url(REDIS_URL)
    gguf_key = f"job:{job_id}:gguf"
    output_dir = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), job_id, "gguf")

    try:
        _publish_deploy_log(r, job_id, f"Exporting GGUF with quantization {quant_type}...")
        from trainer.merge import export_gguf
        gguf_path = export_gguf(merged_model_path, output_dir, quant_type)
        r.set(gguf_key, json.dumps({"status": "done", "gguf_path": gguf_path}))
        _publish_deploy_log(r, job_id, f"GGUF export complete: {os.path.basename(gguf_path)}")
        return gguf_path
    except Exception as e:
        r.set(gguf_key, json.dumps({"status": "failed", "error": str(e)}))
        _publish_deploy_log(r, job_id, f"GGUF export failed: {e}")
        raise


@celery_app.task(bind=True, name="workers.merge_task.push_github")
def push_github_task(self, job_id: str, adapter_path: str, repo_url: str, github_token: str, commit_message: str = "Add fine-tuned LoRA adapter"):
    import subprocess
    import tempfile

    r = redis.from_url(REDIS_URL)

    try:
        _publish_deploy_log(r, job_id, "Pushing adapter to GitHub...")

        # Inject token into URL
        if "https://" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{github_token}@")
        else:
            auth_url = repo_url

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", auth_url, tmp], check=True, capture_output=True)

            # Copy adapter files
            import shutil
            dest = os.path.join(tmp, "adapter")
            shutil.copytree(adapter_path, dest, dirs_exist_ok=True)

            # Set up LFS for large files
            subprocess.run(["git", "-C", tmp, "lfs", "install"], check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "lfs", "track", "*.safetensors"], check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "add", ".gitattributes"], check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "add", "adapter/"], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", tmp, "commit", "-m", commit_message],
                check=True, capture_output=True,
                env={**os.environ, "GIT_AUTHOR_NAME": "TuneOS", "GIT_AUTHOR_EMAIL": "tuneos@bot.local",
                     "GIT_COMMITTER_NAME": "TuneOS", "GIT_COMMITTER_EMAIL": "tuneos@bot.local"},
            )
            subprocess.run(["git", "-C", tmp, "push"], check=True, capture_output=True)

        _publish_deploy_log(r, job_id, f"Pushed adapter to {repo_url}")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode() if e.stderr else str(e)
        _publish_deploy_log(r, job_id, f"GitHub push failed: {msg}")
        raise RuntimeError(msg) from e
    except Exception as e:
        _publish_deploy_log(r, job_id, f"GitHub push failed: {e}")
        raise
