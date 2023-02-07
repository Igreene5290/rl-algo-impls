import argparse
import subprocess
import wandb
import wandb.apis.public

from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import List, NamedTuple


class RunGroup(NamedTuple):
    algo: str
    env_id: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls-benchmarks",
        help="WandB project name to load runs from",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB team of project. None uses default entity",
    )
    parser.add_argument("--wandb-tags", type=str, nargs="+", help="WandB tags")
    parser.add_argument("--wandb-report-url", type=str, help="Link to WandB report")
    parser.add_argument(
        "--envs", type=str, nargs="*", help="Optional filter down to these envs"
    )
    parser.add_argument(
        "--huggingface-user",
        type=str,
        default=None,
        help="Huggingface user or team to upload model cards. Defaults to huggingface-cli login user",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help="How many publish jobs can run in parallel",
    )
    parser.set_defaults(
        wandb_tags=["benchmark_5598ebc", "host_192-9-145-26"],
        wandb_report_url="https://api.wandb.ai/links/sgoodfriend/6p2sjqtn",
        envs=["CartPole-v1", "Acrobot-v1"],
    )
    args = parser.parse_args()
    print(args)

    api = wandb.Api()
    all_runs = api.runs(
        f"{args.wandb_entity or api.default_entity}/{args.wandb_project_name}"
    )

    required_tags = set(args.wandb_tags)
    runs: List[wandb.apis.public.Run] = [
        r
        for r in all_runs
        if required_tags.issubset(set(r.config.get("wandb_tags", [])))
    ]

    runs_paths_by_group = defaultdict(list)
    for r in runs:
        algo = r.config["algo"]
        env = r.config["env"]
        if args.envs and env not in args.envs:
            continue
        run_group = RunGroup(algo, env)
        runs_paths_by_group[run_group].append("/".join(r.path))

    def run(run_paths: List[str]) -> None:
        publish_args = ["python", "huggingface_publish.py"]
        publish_args.append("--wandb-run-paths")
        publish_args.extend(run_paths)
        publish_args.append("--wandb-report-url")
        publish_args.append(args.wandb_report_url)
        if args.huggingface_user:
            publish_args.append("--huggingface-user")
            publish_args.append(args.huggingface_user)
        subprocess.run(publish_args)

    tp = ThreadPool(args.pool_size)
    for run_paths in runs_paths_by_group.values():
        tp.apply_async(run, (run_paths,))
    tp.close()
    tp.join()
