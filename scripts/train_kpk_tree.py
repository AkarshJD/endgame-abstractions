import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    path = "data/processed/kpk/features.csv"

    df = pd.read_csv(path)

    X = df.drop(columns=["wdl"])
    y = df["wdl"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("=== Evaluation ===")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
