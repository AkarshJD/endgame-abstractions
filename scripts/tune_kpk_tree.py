import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


DATA_PATH = "data/processed/kpk/features.csv"


def main():

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Target and features
    X = df.drop(columns=["wdl"])
    y = df["wdl"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building model...")

    tree = DecisionTreeClassifier(random_state=42)

    param_grid = {

        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],

        "min_samples_leaf": [1, 5, 20, 50, 100, 200],

        "min_samples_split": [2, 10, 50, 100],

        "max_features": [None, "sqrt", "log2"]
    }

    grid = GridSearchCV(
        tree,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    print("Running grid search...")
    grid.fit(X_train, y_train)

    print("\n=== Best Parameters ===")
    print(grid.best_params_)

    print("\n=== Best CV Score ===")
    print(grid.best_score_)

    best_model = grid.best_estimator_

    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("\nTree size:")
    print("Depth:", best_model.get_depth())
    print("Leaves:", best_model.get_n_leaves())


if __name__ == "__main__":
    main()
