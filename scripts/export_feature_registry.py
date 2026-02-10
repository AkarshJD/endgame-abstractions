import yaml
import csv

INPUT = "src/endgame/features/feature_registry.yaml"
OUTPUT = "docs/feature_registry.csv"

def csv_safe(text: str) -> str:
    return " ".join(text.split())

with open(INPUT, "r") as f:
    data = yaml.safe_load(f)

features = data["features"]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id",
        "name",
        "type",
        "primitive",
        "pieces",
        "geometry",
        "formula",
        "description",
        "interpretability_level"
    ])

    for feat in features:
        writer.writerow([
            str(feat["id"]),
            feat["name"],
            feat["type"],
            feat["primitive"],
            ",".join(feat["pieces"]),
            feat["geometry"],
            feat["formula"],
            csv_safe(feat.get("description", "")),
            feat.get("interpretability_level", 0),
        ])

print(f"Exported {len(features)} features to {OUTPUT}")
