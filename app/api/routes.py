import time
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.errors import ServiceError
from app.api.schemas import PredictResponse

router = APIRouter()


@router.get("/healthz")
async def health():
    return {"status": "alive"}


@router.get("/readyz")
async def ready(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None or not service.accepting:
        raise ServiceError(503, "not_ready", "服务未就绪或正在停机")
    return {"status": "ready"}


@router.get("/metrics")
async def metrics(request: Request):
    return Response(
        generate_latest(request.app.state.metrics.registry),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@router.get("/v1/models")
async def models(request: Request):
    service, s = request.app.state.service, request.app.state.settings
    return {
        "models": [
            {
                "name": s.model_name,
                "device": service.device,
                "serve_mode": s.serve_mode,
                "max_batch_size": service.max_batch,
                "max_wait_ms": s.max_wait_ms if service.max_batch > 1 else 0,
                "queue_maxsize": service.queue.maxsize,
                "torch_threads": s.thread_count(),
                "ready": service.accepting,
            }
        ]
    }


@router.post("/v1/models/{model}/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    model: str,
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form()] = 5,
):
    s, service = request.app.state.settings, request.app.state.service
    try:
        if model != s.model_name:
            raise ServiceError(400, "unknown_model", "模型名称不存在")
        if not 1 <= top_k <= service.num_classes:
            raise ServiceError(400, "invalid_top_k", f"top_k 必须介于 1 和 {service.num_classes}")
        data = await file.read(s.max_upload_bytes + 1)
        if len(data) > s.max_upload_bytes:
            raise ServiceError(413, "file_too_large", "图片超过大小限制")
        if not data:
            raise ServiceError(400, "invalid_image", "图片不能为空")
        result = await service.predict(data, top_k, request.state.deadline)
    finally:
        await file.close()
    timing = {
        "queue_ms": result.queue_ms,
        "preprocess_ms": result.preprocess_ms,
        "inference_ms": result.inference_ms,
    }
    request.state.timing = timing
    request.state.batch_size = result.batch_size
    return {
        "request_id": request.state.request_id,
        "model": model,
        "predictions": result.predictions,
        "batch_size": result.batch_size,
        "timing": {**timing, "total_ms": (time.monotonic() - request.state.started) * 1000},
    }
