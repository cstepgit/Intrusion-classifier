from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_LABELS = {"normal": 0, "anomaly": 1}


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_data(path: str | Path = "Data/Train_data.csv") -> tuple[pd.DataFrame, pd.Series]:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Follow the setup instructions in README.md."
        )

    data = pd.read_csv(data_path)
    if "class" not in data.columns:
        raise ValueError("Dataset must contain a 'class' target column.")

    labels = data.pop("class").astype(str).str.lower()
    unknown_labels = sorted(set(labels) - TARGET_LABELS.keys())
    if unknown_labels:
        raise ValueError(f"Unsupported target labels: {unknown_labels}")

    return data, labels.map(TARGET_LABELS).astype("int64")


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor(features: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = features.columns.difference(categorical_columns, sort=False).tolist()
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(features, scale_numeric=False)),
            ("model", model),
        ]
    )


def evaluate_predictions(
    target: pd.Series | np.ndarray,
    predictions: np.ndarray,
    *,
    model_name: str,
) -> None:
    print(f"{model_name} test accuracy: {accuracy_score(target, predictions):.6f}")
    print(
        classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            digits=4,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["normal", "anomaly"],
        yticklabels=["normal", "anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} test confusion matrix")
    plt.show()


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)


def aggregate_feature_importance(pipeline: Pipeline) -> pd.Series:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    aggregated: dict[str, float] = {}

    numeric_columns = list(preprocessor.transformers_[0][2])
    numeric_slice = preprocessor.output_indices_["numeric"]
    for column, importance in zip(numeric_columns, importances[numeric_slice], strict=True):
        aggregated[column] = float(importance)

    categorical_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["categorical"]
    categorical_importances = importances[preprocessor.output_indices_["categorical"]]
    offset = 0
    for column, categories in zip(categorical_columns, encoder.categories_, strict=True):
        width = len(categories)
        aggregated[column] = float(categorical_importances[offset : offset + width].sum())
        offset += width

    return pd.Series(aggregated).sort_values(ascending=False)


def select_features_by_importance(
    importances: pd.Series, threshold: float = 0.95
) -> pd.Series:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in the interval (0, 1].")
    normalized = importances.sort_values(ascending=False) / importances.sum()
    count = int(np.searchsorted(normalized.cumsum().to_numpy(), threshold, side="left")) + 1
    return normalized.iloc[:count]


class IntrusionDetectionNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class NeuralNetworkResult:
    model: IntrusionDetectionNN
    preprocessor: ColumnTransformer
    train_losses: list[float]
    validation_losses: list[float]


def set_random_seeds(random_state: int = RANDOM_STATE) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def train_neural_network(
    splits: DataSplits,
    *,
    epochs: int = 1000,
    patience: int = 10,
    learning_rate: float = 0.005,
    random_state: int = RANDOM_STATE,
) -> NeuralNetworkResult:
    set_random_seeds(random_state)
    preprocessor = build_preprocessor(splits.X_train, scale_numeric=True)
    X_train = preprocessor.fit_transform(splits.X_train)
    X_validation = preprocessor.transform(splits.X_validation)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        splits.y_train.to_numpy(), dtype=torch.float32
    ).unsqueeze(1)
    y_validation_tensor = torch.tensor(
        splits.y_validation.to_numpy(), dtype=torch.float32
    ).unsqueeze(1)

    model = IntrusionDetectionNN(X_train_tensor.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(X_train_tensor), y_train_tensor)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(
                model(X_validation_tensor), y_validation_tensor
            ).item()
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}: train loss={train_loss.item():.4f}, "
                f"validation loss={validation_loss:.4f}"
            )
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return NeuralNetworkResult(model, preprocessor, train_losses, validation_losses)


def predict_neural_network(
    result: NeuralNetworkResult, features: pd.DataFrame
) -> np.ndarray:
    transformed = result.preprocessor.transform(features)
    inputs = torch.tensor(transformed, dtype=torch.float32)
    result.model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(result.model(inputs))
    return (probabilities >= 0.5).to(torch.int64).numpy().ravel()


def save_neural_network(result: NeuralNetworkResult, directory: str | Path) -> None:
    artifact_directory = Path(directory)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.preprocessor, artifact_directory / "preprocessor.joblib")
    torch.save(
        {
            "input_size": result.model.network[0].in_features,
            "state_dict": result.model.state_dict(),
            "target_labels": TARGET_LABELS,
        },
        artifact_directory / "model.pt",
    )
