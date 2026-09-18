from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "sensory_score"
COMPOSITION = ["cabbage_pct", "radish_pct", "seasoning_pct", "salt_pct"]
NUMERIC = COMPOSITION + ["brine_salinity_pct", "brining_hours", "fermentation_temp_c", "fermentation_hours", "storage_temp_c"]
CATEGORICAL = ["cabbage_origin", "line"]
FEATURES = NUMERIC + CATEGORICAL

# 데모 전용 임시 범위와 원가지수. 실제 승인 배합표/규격/구매단가로 교체해야 합니다.
LIMITS = {
    "cabbage_pct": (66.0, 76.0), "radish_pct": (5.0, 11.0), "seasoning_pct": (15.0, 23.0), "salt_pct": (1.2, 2.6),
    "brine_salinity_pct": (9.0, 13.0), "brining_hours": (8.0, 16.0), "fermentation_temp_c": (2.0, 8.0),
    "fermentation_hours": (18.0, 72.0), "storage_temp_c": (-1.0, 4.0),
}
COST = {"cabbage_pct": 0.55, "radish_pct": 0.32, "seasoning_pct": 1.55, "salt_pct": 0.12}
DEMO_REVIEW_SCORE = 78.0


def _demo_data(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cabbage = rng.uniform(*LIMITS["cabbage_pct"], rows); radish = rng.uniform(*LIMITS["radish_pct"], rows); salt = rng.uniform(*LIMITS["salt_pct"], rows)
    seasoning = 100 - cabbage - radish - salt
    valid = (seasoning >= LIMITS["seasoning_pct"][0]) & (seasoning <= LIMITS["seasoning_pct"][1])
    while not valid.all():
        n = int((~valid).sum()); cabbage[~valid] = rng.uniform(*LIMITS["cabbage_pct"], n); radish[~valid] = rng.uniform(*LIMITS["radish_pct"], n); salt[~valid] = rng.uniform(*LIMITS["salt_pct"], n)
        seasoning = 100 - cabbage - radish - salt; valid = (seasoning >= LIMITS["seasoning_pct"][0]) & (seasoning <= LIMITS["seasoning_pct"][1])
    salinity = rng.uniform(*LIMITS["brine_salinity_pct"], rows); brining = rng.uniform(*LIMITS["brining_hours"], rows)
    ferment_temp = rng.uniform(*LIMITS["fermentation_temp_c"], rows); ferment_hours = rng.uniform(*LIMITS["fermentation_hours"], rows); storage_temp = rng.uniform(*LIMITS["storage_temp_c"], rows)
    origin = rng.choice(["평창", "강릉", "기타계약"], rows, p=[.68, .20, .12]); line = rng.choice(["K1", "K2"], rows, p=[.58, .42])
    score = (86 - abs(salinity - 11.1) * 2 - abs(brining - 12) * .65 - abs(ferment_temp - 4.8) * 1.15
             - abs(ferment_hours - 42) * .13 - abs(storage_temp - 1) * .8 - abs(seasoning - 19.5) * .45
             + (origin == "평창") * .8 + (line == "K2") * .25 + rng.normal(0, 1.8, rows))
    frame = pd.DataFrame({"lot_id": [f"PC-K-{i+1:04d}" for i in range(rows)], "produced_at": pd.date_range("2025-01-02", periods=rows, freq="12h"),
        "cabbage_pct": cabbage, "radish_pct": radish, "seasoning_pct": seasoning, "salt_pct": salt,
        "brine_salinity_pct": salinity, "brining_hours": brining, "fermentation_temp_c": ferment_temp,
        "fermentation_hours": ferment_hours, "storage_temp_c": storage_temp, "cabbage_origin": origin, "line": line, TARGET: score})
    for col in ["brine_salinity_pct", "fermentation_temp_c", "storage_temp_c"]: frame.loc[rng.choice(rows, 5, replace=False), col] = np.nan
    return frame


def _pipeline(model: Any) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("prepare", ColumnTransformer([("num", numeric, NUMERIC), ("cat", category, CATEGORICAL)])), ("model", model)])


@dataclass
class ModelState:
    data: pd.DataFrame; model: Pipeline; champion: str; metrics: list[dict[str, float | str]]; residual_threshold: float


@lru_cache(maxsize=1)
def state() -> ModelState:
    data = _demo_data(); train, test = train_test_split(data, test_size=.22, random_state=SEED)
    candidates = {"릿지 회귀": Ridge(alpha=1.0), "엑스트라 트리": ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=SEED),
                  "그래디언트 부스팅": GradientBoostingRegressor(random_state=SEED, n_estimators=140, max_depth=2)}
    ranked = []
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator).fit(train[FEATURES], train[TARGET]); pred = pipe.predict(test[FEATURES])
        ranked.append({"model": name, "r2": round(float(r2_score(test[TARGET], pred)), 4), "mae_score": round(float(mean_absolute_error(test[TARGET], pred)), 3), "rmse_score": round(float(sqrt(mean_squared_error(test[TARGET], pred))), 3)})
    ranked.sort(key=lambda item: item["mae_score"]); champion = str(ranked[0]["model"]); final = _pipeline(candidates[champion]).fit(data[FEATURES], data[TARGET])
    residual = np.abs(data[TARGET] - final.predict(data[FEATURES])); return ModelState(data, final, champion, ranked, float(np.quantile(residual, .9)))


def predict(record: dict[str, Any]) -> dict[str, Any]:
    s = state(); value = float(s.model.predict(pd.DataFrame([{key: record[key] for key in FEATURES}]))[0]); total = sum(record[key] for key in COMPOSITION); warnings = []
    if not 99.7 <= total <= 100.3: warnings.append(f"배합 합계가 {total:.2f}%입니다(데모 허용 99.7~100.3%).")
    if value < DEMO_REVIEW_SCORE: warnings.append("예측 점수가 데모 검토 기준보다 낮아 관능·이화학 검사가 필요합니다.")
    return {"prediction_score": round(value, 2), "status": "검토 필요" if warnings else "예측 범위 양호", "warnings": warnings,
            "model": s.champion, "notice": "모델 예측은 관능검사, 이화학검사 또는 출하 승인을 대체하지 않습니다."}


def optimize(target_score: float, max_material_cost: float, cabbage_origin: str, line: str) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + int(target_score * 10) + int(max_material_cost)); candidates = []
    while len(candidates) < 3500:
        cabbage = rng.uniform(*LIMITS["cabbage_pct"]); radish = rng.uniform(*LIMITS["radish_pct"]); salt = rng.uniform(*LIMITS["salt_pct"]); seasoning = 100 - cabbage - radish - salt
        if LIMITS["seasoning_pct"][0] <= seasoning <= LIMITS["seasoning_pct"][1]: candidates.append({"cabbage_pct": cabbage, "radish_pct": radish, "seasoning_pct": seasoning, "salt_pct": salt})
    frame = pd.DataFrame(candidates)
    for key in ["brine_salinity_pct", "brining_hours", "fermentation_temp_c", "fermentation_hours", "storage_temp_c"]: frame[key] = rng.uniform(*LIMITS[key], len(frame))
    frame["cabbage_origin"], frame["line"] = cabbage_origin, line; frame["prediction"] = state().model.predict(frame[FEATURES]); frame["cost"] = sum(frame[key] * value for key, value in COST.items())
    frame["violation"] = np.maximum(0, target_score - frame["prediction"]) / 3 + np.maximum(0, frame["cost"] - max_material_cost) / 5
    feasible = frame[(frame.prediction >= target_score) & (frame.cost <= max_material_cost)]
    chosen = feasible.sort_values(["cost", "prediction"], ascending=[True, False]).iloc[0] if len(feasible) else frame.sort_values(["violation", "cost"]).iloc[0]
    unmet = []
    if chosen.prediction < target_score: unmet.append(f"목표 품질점수 {target_score:.1f}점 미달")
    if chosen.cost > max_material_cost: unmet.append("원재료비 지수 상한 초과")
    return {"feasible": bool(len(feasible)), "prediction_score": round(float(chosen.prediction), 2), "material_cost_index": round(float(chosen.cost), 2),
            "settings": {key: round(float(chosen[key]), 3) for key in NUMERIC}, "context": {"cabbage_origin": cabbage_origin, "line": line},
            "unmet_constraints": unmet, "model": state().champion, "notice": "데모 탐색 결과이며 승인 배합표 확인과 시험 생산 후 적용해야 합니다."}

