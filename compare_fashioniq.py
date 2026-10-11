import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = {
    "No negative sampling (30 epochs, batch 8)": (
        ROOT / "cir_eval/fiq/new_qformer_2026-10-09_0.json"
    ),
    "Original code (5 epochs, batch 32)": (
        ROOT / "cir_eval/fiq/debug_2026-10-08_0.json"
    ),
    "Original code (30 epochs, batch 8)": (
        ROOT / "cir_eval/fiq/debug_2026-10-10_4.json"
    ),
}
CATEGORIES = ("dress", "shirt", "toptee", "average")
RECALL_KS = (5, 10, 50)


def load_results(path):
    with path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def metric_key(category, k):
    if category == "average":
        return f"average_recall_at{k}"
    return f"fashionIQ_{category}_recall@{k}"


def build_rows(results, baseline):
    rows = []
    for category in CATEGORIES:
        for k in RECALL_KS:
            values = {
                label: data[metric_key(category, k)]
                for label, data in results.items()
            }
            rows.append(
                {
                    "Category": category.capitalize(),
                    "Metric": f"R@{k}",
                    **values,
                    "Delta vs 5-epoch baseline (pp)": (
                        values["Original code (30 epochs, batch 8)"]
                        - values[baseline]
                    ),
                }
            )
    return rows


def print_table(rows, result_labels):
    columns = ("Category", "Metric", *result_labels, "Delta vs 5-epoch baseline (pp)")
    widths = {
        column: max(
            len(column),
            *(len(str(row[column])) if column in ("Category", "Metric") else 0 for row in rows),
        )
        for column in columns
    }
    for row in rows:
        for column in result_labels:
            widths[column] = max(widths[column], len(f"{row[column]:.2f}"))
        widths["Delta vs 5-epoch baseline (pp)"] = max(
            widths["Delta vs 5-epoch baseline (pp)"],
            len(f"{row['Delta vs 5-epoch baseline (pp)']:+.2f}"),
        )

    def format_row(values):
        cells = []
        for column, value in zip(columns, values):
            if column in result_labels:
                cell = f"{value:.2f}"
            elif column == "Delta vs 5-epoch baseline (pp)":
                cell = f"{value:+.2f}"
            else:
                cell = str(value)
            cells.append(cell.ljust(widths[column]))
        return " | ".join(cells)

    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(format_row([row[column] for column in columns]))


def main():
    parser = argparse.ArgumentParser(
        description="Compare FashionIQ Recall@5/10/50 across three training runs."
    )
    parser.add_argument(
        "--no-negative",
        type=Path,
        default=DEFAULT_RESULTS["No negative sampling (30 epochs, batch 8)"],
        help="Evaluation JSON for the run without negative sampling.",
    )
    parser.add_argument(
        "--original-5-epoch",
        type=Path,
        default=DEFAULT_RESULTS["Original code (5 epochs, batch 32)"],
        help="Evaluation JSON for the original 5-epoch, batch-32 run.",
    )
    parser.add_argument(
        "--original-30-epoch",
        type=Path,
        default=DEFAULT_RESULTS["Original code (30 epochs, batch 8)"],
        help="Evaluation JSON for the original 30-epoch, batch-8 run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "cir_eval/fiq/comparison_3_runs.csv",
        help="Path for the CSV comparison table.",
    )
    args = parser.parse_args()

    paths = {
        "No negative sampling (30 epochs, batch 8)": args.no_negative,
        "Original code (5 epochs, batch 32)": args.original_5_epoch,
        "Original code (30 epochs, batch 8)": args.original_30_epoch,
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Cannot compare yet; evaluation result JSON is missing:\n"
            f"{missing_list}\n"
            "Run FashionIQ evaluation for the missing checkpoint first."
        )

    results = {label: load_results(path) for label, path in paths.items()}
    baseline = "Original code (5 epochs, batch 32)"
    rows = build_rows(results, baseline)

    print_table(rows, tuple(paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV comparison to: {args.output}")


if __name__ == "__main__":
    main()
