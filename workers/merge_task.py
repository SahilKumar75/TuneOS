"""Celery tasks for post-training operations: merge, GGUF export, GitHub push."""

import json
import os

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
def _run_merge_impl(
    job_id: str, base_model_id: str, adapter_path: str, output_path: str, hf_token: str = ""
):
    from trainer.merge import merge_adapter

    return merge_adapter(base_model_id, adapter_path, output_path, hf_token)


@celery_app.task(bind=True, name="workers.merge_task.merge_adapter")
def merge_adapter_task(self, job_id: str, base_model_id: str, adapter_path: str):
    r = redis.from_url(REDIS_URL)
    _tok = r.getdel(f"job:{job_id}:hf_token")
    hf_token = (_tok.decode() if isinstance(_tok, bytes) else _tok) or ""
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
def push_github_task(
    self,
    job_id: str,
    adapter_path: str,
    repo_url: str,
    commit_message: str = "Add fine-tuned LoRA adapter",
):
    import subprocess
    import tempfile

    r = redis.from_url(REDIS_URL)
    _tok = r.getdel(f"job:{job_id}:github_token")
    github_token = (_tok.decode() if isinstance(_tok, bytes) else _tok) or ""

    # Only allow GitHub HTTPS remotes — reject arbitrary hosts
    if not repo_url.startswith("https://github.com/"):
        raise ValueError(f"Only https://github.com/ remotes are supported, got: {repo_url}")

    try:
        _publish_deploy_log(r, job_id, "Pushing adapter to GitHub...")

        # Provide the token via a credential helper so it never appears in
        # command args, process listings, or error output.
        import stat
        import textwrap

        with tempfile.TemporaryDirectory() as tmp:
            # Write a one-shot credential helper script
            helper_path = os.path.join(tmp, "git-credential-tuneos")
            helper_script = textwrap.dedent(f"""\
                #!/bin/sh
                echo username=x-token
                echo password={github_token}
            """)
            with open(helper_path, "w") as fh:
                fh.write(helper_script)
            os.chmod(helper_path, stat.S_IRWXU)

            clone_env = {
                **os.environ,
                "GIT_ASKPASS": helper_path,
                "GIT_TERMINAL_PROMPT": "0",
            }

            repo_dir = os.path.join(tmp, "repo")
            subprocess.run(
                ["git", "clone", repo_url, repo_dir], check=True, capture_output=True, env=clone_env
            )

            # Copy adapter files into the cloned repo
            import shutil

            dest = os.path.join(repo_dir, "adapter")
            shutil.copytree(adapter_path, dest, dirs_exist_ok=True)

            # Set up LFS for large files
            subprocess.run(
                ["git", "-C", repo_dir, "lfs", "install"], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", repo_dir, "lfs", "track", "*.safetensors"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", repo_dir, "add", ".gitattributes"], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", repo_dir, "add", "adapter/"], check=True, capture_output=True
            )
            subprocess.run(
                ["git", "-C", repo_dir, "commit", "-m", commit_message],
                check=True,
                capture_output=True,
                env={
                    **clone_env,
                    "GIT_AUTHOR_NAME": "TuneOS",
                    "GIT_AUTHOR_EMAIL": "tuneos@bot.local",
                    "GIT_COMMITTER_NAME": "TuneOS",
                    "GIT_COMMITTER_EMAIL": "tuneos@bot.local",
                },
            )
            subprocess.run(
                ["git", "-C", repo_dir, "push"],
                check=True,
                capture_output=True,
                env=clone_env,
            )

        _publish_deploy_log(r, job_id, f"Pushed adapter to {repo_url}")
    except subprocess.CalledProcessError as e:
        # Strip any token that might appear in error output before logging
        raw_msg = e.stderr.decode() if e.stderr else str(e)
        safe_msg = raw_msg.replace(github_token, "***") if github_token else raw_msg
        _publish_deploy_log(r, job_id, f"GitHub push failed: {safe_msg}")
        raise RuntimeError(safe_msg) from e
    except Exception as e:
        safe_e = str(e).replace(github_token, "***") if github_token else str(e)
        _publish_deploy_log(r, job_id, f"GitHub push failed: {safe_e}")
        raise
