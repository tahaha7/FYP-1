import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

# subset name -> folder name in the repo
# the real repo has corpus/ for training data and benchmark/ for the
# manually checked set, the dataset card shows a different layout
SUBSETS = {
    "CS": "US-CS",
    "Std": "US-Std",
    "EngPk": "US-EngPk",
}

CATEGORIES = [
    "comedy_show", "drama", "film", "food", "interview", "news",
    "podcast", "poetry", "proses", "roadside_interview", "vlogs",
    "youtube_review",
]

# benchmark folder uses different category names, uppercase and shortened
BENCH_CATEGORIES = {
    "comedy_show": "COMEDYSHOW",
    "drama": "DRAMA",
    "film": "FILM",
    "food": "FOOD",
    "interview": "INTERVIEWS",
    "news": "NEWS",
    "podcast": "PODCAST",
    "poetry": "POETRY",
    "proses": "PROSES",
    "roadside_interview": "ROADSIDE",
    "vlogs": "VLOG",
    "youtube_review": "YOUTUBEREVIEWS",
}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_patterns(subset, fmt, categories, benchmark=False):
    base = SUBSETS[subset]
    patterns = []
    for cat in categories:
        if benchmark:
            bench_cat = BENCH_CATEGORIES[cat]
            patterns.append(f"benchmark/US-benchmark-{subset}/{fmt}/{bench_cat}/**")
            # podcast is split across two folders in the benchmark, messy
            if cat == "podcast":
                patterns.append(f"benchmark/US-benchmark-{subset}/{fmt}/Podcast/**")
        else:
            patterns.append(f"corpus/{base}/{fmt}/{cat}/**")
    return patterns


def local_folder(subset, fmt, cat, benchmark=False):
    if benchmark:
        return Path(f"benchmark/US-benchmark-{subset}") / fmt / BENCH_CATEGORIES[cat]
    return Path("corpus") / SUBSETS[subset] / fmt / cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="CS", choices=list(SUBSETS))
    ap.add_argument("--format", default="short", choices=["short", "long"])
    ap.add_argument("--category", default="interview",
                    help="one category, or 'all'")
    ap.add_argument("--max-categories", type=int, default=1,
                    help="used when --category all, takes first N categories")
    ap.add_argument("--benchmark", action="store_true",
                    help="download the manually checked benchmark set instead of corpus")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="override download folder")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_id = cfg["dataset"]["repo_id"]
    # benchmark goes to its own folder by default, so it can never be mixed
    # into the training manifests by accident
    if args.out:
        out_dir = Path(args.out)
    elif args.benchmark:
        out_dir = Path(cfg["dataset"].get("benchmark_dir", "data/benchmark_raw"))
    else:
        out_dir = Path(cfg["dataset"]["local_dir"])

    if args.category == "all":
        cats = CATEGORIES[:args.max_categories]
    else:
        cats = [args.category]

    patterns = build_patterns(args.subset, args.format, cats, args.benchmark)
    kind = "benchmark" if args.benchmark else "corpus"
    print(f"downloading {repo_id}: {kind} subset={args.subset} format={args.format} categories={cats}")
    print("this can take a while if you picked a lot of categories")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=patterns,
    )

    print(f"done. files are in: {path}")
    for cat in cats:
        folder = Path(path) / local_folder(args.subset, args.format, cat, args.benchmark)
        print(f"  {cat}: {folder} (exists: {folder.exists()})")


if __name__ == "__main__":
    main()
