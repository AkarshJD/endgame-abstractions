import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    df = pd.read_csv("data/processed/kpk/features.csv")

    X = df.drop(columns=["wdl", "dtz"])
    y = df["wdl"]

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=200,
        random_state=42
    )

    model.fit(X, y)

    rules = export_text(
        model,
        feature_names=list(X.columns)
    )

    print(rules)


if __name__ == "__main__":
    main()
