from dataclasses import dataclass, field

@dataclass()
class TrainingParams:
        model_type: str = field(default="RandomForest")
        n_estimators: int = field(default=100)
        random_state: int = field(default=255)
    