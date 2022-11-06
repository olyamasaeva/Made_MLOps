from dataclasses import dataclass, field
import json

@dataclass()
class MetricsClass:
    accuracy: float = field(default=0)
    precision: float = field(default=0)
    recall: float = field(default=0)
    f1_score: float = field(default=0)
    roc_auc_score: float = field(default=0)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
