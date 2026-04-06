"""
inference.py - CDN Cache Optimizer Baseline Agent
Uses OpenAI client to run an LLM agent against the environment.
Emits structured [START], [STEP], [END] logs to stdout.

Required env vars:
  API_BASE_URL  - LLM API endpoint
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face / API key
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI
from env.cache import CDNCacheEnv, TASK_CONFIGS
from env.models import Action, Observation

# ─────────────────────────────────────────────
# Config from environment (required by OpenEnv spec)
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set. Using API_BASE_URL without auth header override.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "placeholder",
)

TASKS = ["task_easy", "task_medium", "task_hard"]
SEED  = 42

# ─────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent CDN cache management agent.

At each step you receive the current cache state and an incoming file request.
Your job: decide which file to evict (if any) to make room for new content.

Rules:
- Only evict a file if the cache is nearly full and the incoming file is NOT already cached
- Prefer evicting files with LOW request_frequency and NOT viral
- Never evict a file that was just evicted (cache thrashing)
- If cache has space, respond with null (no eviction needed)

You MUST respond with ONLY valid JSON in this exact format:
{"evict_file_id": "<file_id>" or null}

No explanation. No markdown. Only the JSON object."""


def build_user_prompt(obs: Observation) -> str:
    cached_summary = []
    for f in obs.cached_files:
        cached_summary.append(
            f"  - {f.file_id}: size={f.size_mb}MB freq={f.request_frequency:.1f} "
            f"viral={f.is_viral} last_accessed=step_{f.last_accessed}"
        )
    cached_str = "\n".join(cached_summary) if cached_summary else "  (empty)"

    space_needed = obs.incoming_file_size_mb
    space_free   = obs.cache_capacity_mb - obs.cache_used_mb

    return f"""Step {obs.step} | Time of day: {obs.time_of_day:.2f} | Hit rate: {obs.recent_hit_rate:.2f}

Cache: {obs.cache_used_mb:.1f}MB / {obs.cache_capacity_mb:.1f}MB used ({obs.cache_fill_ratio*100:.1f}% full)
Free space: {space_free:.1f}MB

Incoming request:
  file_id: {obs.incoming_file_id}
  size: {obs.incoming_file_size_mb}MB
  viral: {obs.incoming_file_is_viral}
  already_cached: {obs.cache_hit}
  space_needed_to_cache: {"none (fits)" if space_free >= space_needed else f"{space_needed - space_free:.1f}MB deficit"}

Next 3 requests preview: {obs.queue_preview}

Currently cached files ({len(obs.cached_files)} files):
{cached_str}

Decide: which file to evict? (null if no eviction needed)"""


def llm_action(obs: Observation, step_num: int) -> Action:
    """Call LLM and parse action. Fall back to LRU on failure."""
    prompt = build_user_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return Action(evict_file_id=parsed.get("evict_file_id"))
    except Exception as e:
        # Fallback: LRU
        if obs.cached_files:
            lru = min(obs.cached_files, key=lambda f: f.last_accessed)
            return Action(evict_file_id=lru.file_id)
        return Action(evict_file_id=None)


# ─────────────────────────────────────────────
# Run one task episode
# ─────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    config = TASK_CONFIGS[task_id]
    env    = CDNCacheEnv(task_id=task_id, seed=SEED)
    obs    = env.reset()

    total_reward = 0.0
    step_num     = 0

    # ── [START] ──
    print(json.dumps({
        "type":    "START",
        "task_id": task_id,
        "task_name": config.name,
        "difficulty": config.difficulty,
        "episode_length": config.episode_length,
        "cache_capacity_mb": config.cache_capacity_mb,
        "model": MODEL_NAME,
        "seed": SEED,
    }))
    sys.stdout.flush()

    while True:
        action = llm_action(obs, step_num)
        result = env.step(action)

        total_reward += result.reward.total

        # ── [STEP] ──
        print(json.dumps({
            "type":           "STEP",
            "task_id":        task_id,
            "step":           step_num,
            "action":         {"evict_file_id": action.evict_file_id},
            "cache_hit":      result.observation.cache_hit,
            "reward":         result.reward.total,
            "reward_breakdown": {
                "cache_hit_bonus":       result.reward.cache_hit_bonus,
                "eviction_penalty":      result.reward.eviction_penalty,
                "thrash_penalty":        result.reward.thrash_penalty,
                "bandwidth_saved":       result.reward.bandwidth_saved,
                "wasted_capacity_penalty": result.reward.wasted_capacity_penalty,
            },
            "cumulative_reward": round(total_reward, 4),
            "hit_rate":       result.observation.recent_hit_rate,
            "cache_fill":     result.observation.cache_fill_ratio,
            "done":           result.done,
        }))
        sys.stdout.flush()

        obs      = result.observation
        step_num += 1

        if result.done:
            break

    final_state = env.state()
    final_hit_rate = final_state["hit_rate"]

    # ── [END] ──
    print(json.dumps({
        "type":              "END",
        "task_id":           task_id,
        "task_name":         config.name,
        "total_steps":       step_num,
        "total_reward":      round(total_reward, 4),
        "final_hit_rate":    round(final_hit_rate, 4),
        "bandwidth_saved_mb": round(final_state["bandwidth_saved_mb"], 2),
        "total_hits":        final_state["hits"],
        "total_misses":      final_state["misses"],
        "score":             round(min(1.0, final_hit_rate / {"task_easy": 0.60, "task_medium": 0.55, "task_hard": 0.45}[task_id]), 4),
    }))
    sys.stdout.flush()

    return {
        "task_id":        task_id,
        "total_reward":   round(total_reward, 4),
        "final_hit_rate": round(final_hit_rate, 4),
        "score":          round(min(1.0, final_hit_rate / {"task_easy": 0.60, "task_medium": 0.55, "task_hard": 0.45}[task_id]), 4),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Starting CDN Cache Optimizer inference", file=sys.stderr)
    print(f"[INFO] Model: {MODEL_NAME} | API: {API_BASE_URL}", file=sys.stderr)

    results = []
    for task_id in TASKS:
        print(f"\n[INFO] Running {task_id}...", file=sys.stderr)
        r = run_task(task_id)
        results.append(r)
        print(f"[INFO] {task_id} done | score={r['score']} hit_rate={r['final_hit_rate']}", file=sys.stderr)

    print("\n[INFO] === FINAL RESULTS ===", file=sys.stderr)
    for r in results:
        print(f"[INFO] {r['task_id']}: score={r['score']} reward={r['total_reward']}", file=sys.stderr)

    overall = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"[INFO] Overall score: {overall}", file=sys.stderr)