from __future__ import annotations

import math
from typing import Literal, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from .ml_service import COMPOSITION, DEMO_REVIEW_SCORE, FEATURES, LIMITS, TARGET, optimize, predict, state

app = FastAPI(title="평창꽃순이김치 제조 ML AI", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

class ProcessInput(BaseModel):
    cabbage_pct: float = Field(ge=55, le=85); radish_pct: float = Field(ge=0, le=20); seasoning_pct: float = Field(ge=8, le=35); salt_pct: float = Field(ge=.5, le=5)
    brine_salinity_pct: float = Field(ge=5, le=18); brining_hours: float = Field(ge=3, le=24); fermentation_temp_c: float = Field(ge=-2, le=15)
    fermentation_hours: float = Field(ge=6, le=120); storage_temp_c: float = Field(ge=-5, le=10)
    cabbage_origin: Literal["평창", "강릉", "기타계약"]; line: Literal["K1", "K2"]
    @model_validator(mode="after")
    def composition(self):
        if not 98 <= sum(getattr(self, key) for key in COMPOSITION) <= 102: raise ValueError("배추·무·양념·소금 배합 합계는 98~102% 범위여야 합니다")
        return self

class OptimizeInput(BaseModel):
    target_score: float = Field(ge=65, le=95); max_material_cost: float = Field(ge=45, le=100)
    cabbage_origin: Literal["평창", "강릉", "기타계약"] = "평창"; line: Literal["K1", "K2"] = "K1"

@app.get("/api/health")
def health(): return {"status": "ok", "model_ready": bool(state().champion), "data_mode": "demo", "product": "평창꽃순이김치"}

@app.get("/api/ml/overview")
def ml_overview():
    s = state(); return {"data_mode": "demo", "rows": len(s.data), "missing_cells": int(s.data.isna().sum().sum()), "duplicate_lots": int(s.data.lot_id.duplicated().sum()),
        "features": FEATURES, "target": {"name": TARGET, "unit": "점"}, "champion": s.champion, "metrics": s.metrics, "limits": LIMITS,
        "warning": "합성 데모 데이터의 성능이며 실제 김치 품질 성능이 아닙니다."}

@app.post("/api/ml/optimize")
def ml_optimize(payload: OptimizeInput): return optimize(**payload.model_dump())

@app.post("/api/quality/predict")
def quality_predict(payload: ProcessInput): return predict(payload.model_dump())

@app.get("/api/quality/overview")
def quality_overview():
    s = state(); pred = s.model.predict(s.data[FEATURES]); residual = abs(s.data[TARGET].to_numpy() - pred); order = residual.argsort()[::-1][:8]
    queue = [{"lot_id": s.data.iloc[i].lot_id, "measured_score": round(float(s.data.iloc[i][TARGET]), 2), "predicted_score": round(float(pred[i]), 2), "residual_score": round(float(residual[i]), 2)} for i in order]
    return {"evaluated": len(s.data), "demo_review_score": DEMO_REVIEW_SCORE, "residual_review_threshold": round(s.residual_threshold, 2), "priority_queue": queue,
            "notice": "잔차 우선순위는 불량 판정이 아니라 데이터/모델 검토 순서입니다."}

@app.get("/api/data/records")
def records(limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0), line: Optional[Literal["K1", "K2"]] = None,
            cabbage_origin: Optional[Literal["평창", "강릉", "기타계약"]] = None, search: Optional[str] = Query(None, max_length=30)):
    frame = state().data.copy()
    if line: frame = frame[frame.line == line]
    if cabbage_origin: frame = frame[frame.cabbage_origin == cabbage_origin]
    if search: frame = frame[frame.lot_id.str.contains(search, case=False, regex=False)]
    frame = frame.sort_values(["produced_at", "lot_id"], ascending=[False, True]); page = frame.iloc[offset:offset+limit]; items = []
    for row in page.to_dict("records"):
        items.append({key: (None if isinstance(value, float) and not math.isfinite(value) else value.isoformat() if hasattr(value, "isoformat") else value) for key, value in row.items()})
    return {"total": len(frame), "limit": limit, "offset": offset, "records": items}

