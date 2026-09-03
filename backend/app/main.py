import logging
import os

# Load backend/.env before other app modules read configuration.
from app import config as _config  # noqa: F401
from app.config import load_backend_env

load_backend_env()

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.course_id import assert_valid_course_id
from app.course_index import build_course_rag_index
from app.course_model_resolution import resolve_current_course_model
from app.course_rag import generate_course_rag_answer
from app.db_routes import router as db_router
from app.training_queue_routes import router as training_queue_router
from app.finetuned_client import (
    check_finetuned_service_health,
    public_service_health,
    generate_finetuned_response,
)
from app.finetuned_rag import generate_course_finetuned_rag_answer
from app.ollama import generate_base_model_response
from app.schemas import (
    BaseModelGenerateRequest,
    EnqueueTrainingRunRequest,
    EnqueueTrainingRunResponse,
    RetryTrainingRunResponse,
    BaseModelGenerateResponse,
    FineTunedGenerateRequest,
    FineTunedGenerateResponse,
    FineTunedHealthResponse,
    FineTunedRagGenerateRequest,
    FineTunedRagGenerateResponse,
    CourseChunkMetadata,
    CourseChunksResponse,
    CourseSeedListResponse,
    FactAllocationCappedFact,
    FactAllocationItem,
    FactAllocationRankingItem,
    FactAllocationRequest,
    FactAllocationResponse,
    FactAllocationSkippedFact,
    FactAllocationSummary,
    FactInventoryItem,
    FactInventoryRequest,
    FactInventoryResponse,
    GeneratedSeedExample,
    ApprovedExportStatusResponse,
    PrepareTrainingSplitRequest,
    PrepareTrainingSplitResponse,
    RagGenerateRequest,
    RagGenerateResponse,
    RagGenerateSource,
    RagRetrieveResult,
    SeedExportApprovedRequest,
    SeedExportApprovedResponse,
    SeedGenerateRequest,
    SeedGenerateResponse,
    SeedQualityCheckRequest,
    SeedQualityCheckResponse,
    SeedReviewRecord,
    SeedReviewRequest,
    SeedReviewResponse,
    StarterSeedGenerateRequest,
    StarterSeedGenerateResponse,
    StarterSeedPersistence,
    StarterSeedProgress,
    StarterSeedTopUpRequest,
    StarterGenerationStatusResponse,
    SyllabusTextResponse,
    SyllabusUploadResponse,
    TrainingLaunchCapabilityResponse,
    TrainingLaunchRequest,
    TrainingLaunchResponse,
)
from app.db import db_connection, translate_db_errors
from app.db_courses import course_exists
from app.db_seeds import list_seeds, review_seed
from app.db_training_runs import (
    ActiveTrainingRunError,
    enqueue_training_run,
)
from app.training_retry import RetryNotEligibleError, retry_training_run
from app.starter_status import reconcile_starter_seed_generation
from app.seed_dataset_quality import inspect_seed_dataset
from app.provenance_privacy import public_training_run
from app.export_privacy import (
    public_export_status,
    public_export_summary,
    public_message,
    public_snapshot_ref,
)
from app.seed_export import FinetuneJsonlValidationError, export_approved_seeds
from app.seed_split import (
    DEFAULT_SPLIT_SEED,
    TrainingSplitError,
    approved_export_status,
    prepare_training_split,
)
from app.seed_review import REVIEW_STATUSES, resolve_review_status
from app.seed_allocation import allocate_slots
from app.seed_generation import generate_seeds_from_chunk, generate_starter_seeds_for_course
from app.fact_inventory_cache import load_or_build_fact_inventory
from app.ollama_coordination import (
    get_starter_job_status,
    starter_job_slot,
)
from app.starter_jobs import (
    run_auto_starter_seed_generation,
    try_queue_starter_seed_generation,
)
from app.storage import get_course_artifact_storage
from app.syllabus_extract import extract_clean_syllabus_text
from app.syllabus_upload import SyllabusUploadError, validate_syllabus_upload
from app.training_launch import (
    LaunchDisabledError,
    LaunchExecutionError,
    LaunchValidationError,
    describe_capability,
    launch_training,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Syllabus Model Lab Backend")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The application's persistence routes. PostgreSQL is the system of record for
# everything the browser reads or writes, and these are what it talks to.
# Importing the router does not open a connection — an unset DATABASE_URL only
# surfaces when an /api/db route is actually called.
app.include_router(db_router)

# The queue the Tillicum runner claims work from. Separate router because it
# has a different caller and a different credential: a shared worker token
# rather than the browser's ordinary access. See training_queue_routes.
app.include_router(training_queue_router)


# Inference and health endpoints are served under BOTH their original paths
# and an /api-prefixed alias.
#
# The deployed Nginx forwards only `location /api/` to uvicorn, so a route
# mounted at the root — `/health`, `/rag/generate` — is unreachable from a
# browser no matter what the frontend composes: `/health` hits the SPA, and
# `/api/health` did not exist. The alias is the smallest fix that does not
# require touching Nginx or the deployed VITE_API_BASE_URL.
#
# The original paths stay. Systemd health checks, the fine-tuned tunnel helper,
# and local `curl localhost:8001/health` all use them, and this backend is
# reachable directly on the VM as well as through the proxy.
@app.get("/api/health")
@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "syllabus-model-lab-backend",
    }


@app.post("/api/base-model/generate", response_model=BaseModelGenerateResponse)
@app.post("/base-model/generate", response_model=BaseModelGenerateResponse)
async def generate_base_model(
    request: BaseModelGenerateRequest,
) -> BaseModelGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        safe_course_id = assert_valid_course_id(request.course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # courseId is validated for route isolation but unused by the base model itself.
    result = await generate_base_model_response(question)
    return BaseModelGenerateResponse(
        answer=result["answer"],
        model=result["model"],
        responseType=result["response_type"],
        courseId=safe_course_id,
    )


@app.get("/api/fine-tuned/health", response_model=FineTunedHealthResponse)
@app.get("/fine-tuned/health", response_model=FineTunedHealthResponse)
async def fine_tuned_health() -> FineTunedHealthResponse:
    # The probe keeps the exact hostname, port and tunnel URL — this route does
    # not hand them to a browser. See `finetuned_client.public_service_health`.
    result = public_service_health(await check_finetuned_service_health())
    return FineTunedHealthResponse(
        status=result["status"],
        model=result.get("model"),
        adapterLoaded=result.get("adapterLoaded"),
        courses=result.get("courses") or [],
        secondsRemaining=result.get("secondsRemaining"),
    )


@app.post("/api/fine-tuned/generate", response_model=FineTunedGenerateResponse)
@app.post("/fine-tuned/generate", response_model=FineTunedGenerateResponse)
async def generate_fine_tuned(
    request: FineTunedGenerateRequest,
) -> FineTunedGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        safe_course_id = assert_valid_course_id(request.course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The course's own model, resolved from PostgreSQL before the cluster is
    # asked anything. This is what makes a fine-tuned answer attributable: the
    # version travels with the request and comes back on the response, and the
    # client refuses a response that names a different course.
    resolved = resolve_current_course_model(safe_course_id)
    result = await generate_finetuned_response(
        question,
        course_id=safe_course_id,
        model_version=resolved["version"],
    )
    return FineTunedGenerateResponse(
        answer=result["answer"],
        model=result["model"],
        responseType=result["response_type"],
        courseId=safe_course_id,
        modelVersion=result.get("model_version") or resolved["version"],
        adapterLoaded=result["adapter_loaded"],
        generationSeconds=result["generation_seconds"],
    )


@app.post("/api/rag/generate", response_model=RagGenerateResponse)
@app.post("/rag/generate", response_model=RagGenerateResponse)
async def generate_rag_response(request: RagGenerateRequest) -> RagGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    result = await generate_course_rag_answer(
        course_id=request.course_id,
        question=question,
        top_k=request.top_k,
    )

    return RagGenerateResponse(
        courseId=result["courseId"],
        answer=result["answer"],
        model=result["model"],
        sources=[RagGenerateSource(**source) for source in result["sources"]],
        retrievedChunks=[
            RagRetrieveResult(**chunk) for chunk in result["retrievedChunks"]
        ],
        responseType=result["responseType"],
    )


@app.post("/api/fine-tuned-rag/generate", response_model=FineTunedRagGenerateResponse)
@app.post("/fine-tuned-rag/generate", response_model=FineTunedRagGenerateResponse)
async def generate_fine_tuned_rag(
    request: FineTunedRagGenerateRequest,
) -> FineTunedRagGenerateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    result = await generate_course_finetuned_rag_answer(
        course_id=request.course_id,
        question=question,
        top_k=request.top_k,
    )

    return FineTunedRagGenerateResponse(
        courseId=result["courseId"],
        answer=result["answer"],
        model=result["model"],
        sources=[RagGenerateSource(**source) for source in result["sources"]],
        retrievedChunks=[
            RagRetrieveResult(**chunk) for chunk in result["retrievedChunks"]
        ],
        responseType=result["responseType"],
        modelVersion=result.get("modelVersion"),
        adapterLoaded=result["adapterLoaded"],
        generationSeconds=result["generationSeconds"],
    )


@app.post(
    "/api/courses/{course_id}/syllabus",
    response_model=SyllabusUploadResponse,
    status_code=201,
)
async def upload_course_syllabus(
    course_id: str,
    background_tasks: BackgroundTasks,
    syllabus_file: UploadFile = File(...),
) -> SyllabusUploadResponse:
    storage = get_course_artifact_storage()

    try:
        validated = await validate_syllabus_upload(course_id, syllabus_file)
    except SyllabusUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    finally:
        await syllabus_file.close()

    try:
        storage.save_original_syllabus(
            validated.course_id,
            validated.syllabus_type,
            validated.content,
        )
        extracted_text = extract_clean_syllabus_text(validated)
        storage.save_extracted_text(validated.course_id, extracted_text)
        index_data = await build_course_rag_index(
            course_id=validated.course_id,
            source_file=validated.original_filename,
            syllabus_text=extracted_text,
            storage=storage,
        )
    except SyllabusUploadError as exc:
        try:
            storage.delete_partial_files(validated.course_id)
        except Exception:
            pass
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001 - map unexpected storage failures to HTTP 500
        try:
            storage.delete_partial_files(validated.course_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Could not save, extract, or index the syllabus file.",
        ) from exc

    queue_result = await try_queue_starter_seed_generation(validated.course_id)
    if queue_result.get("queued"):
        background_tasks.add_task(
            run_auto_starter_seed_generation,
            validated.course_id,
        )

    return SyllabusUploadResponse(
        courseId=validated.course_id,
        syllabusFileName=validated.original_filename,
        syllabusType=validated.syllabus_type,
        syllabusStatus="indexed",
        fileSize=validated.file_size,
        characterCount=len(extracted_text),
        chunkCount=int(index_data["chunkCount"]),
        starterSeedGenerationStatus=queue_result.get("status"),
    )


@app.get(
    "/api/courses/{course_id}/syllabus/text",
    response_model=SyllabusTextResponse,
)
def get_course_syllabus_text(course_id: str) -> SyllabusTextResponse:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    storage = get_course_artifact_storage()
    text = storage.load_extracted_text(safe_course_id)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail="Extracted syllabus text was not found for this course.",
        )

    return SyllabusTextResponse(
        courseId=safe_course_id,
        text=text,
        characterCount=len(text),
    )


@app.get(
    "/api/courses/{course_id}/chunks",
    response_model=CourseChunksResponse,
)
def get_course_chunks(course_id: str) -> CourseChunksResponse:
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    chunks = [
        CourseChunkMetadata(
            chunkId=chunk["chunkId"],
            sectionTitle=chunk["sectionTitle"],
            text=chunk["text"],
            order=chunk["order"],
            documentTitle=chunk.get("documentTitle"),
            headingPath=chunk.get("headingPath"),
            startOffset=chunk.get("startOffset"),
            endOffset=chunk.get("endOffset"),
        )
        for chunk in index_data.get("chunks", [])
    ]

    return CourseChunksResponse(
        courseId=safe_course_id,
        chunkCount=len(chunks),
        chunks=chunks,
        indexVersion=index_data.get("indexVersion"),
        documentTitle=index_data.get("documentTitle"),
    )


@app.post(
    "/api/courses/{course_id}/seeds/generate",
    response_model=SeedGenerateResponse,
)
async def generate_course_seeds(
    course_id: str,
    request: SeedGenerateRequest,
) -> SeedGenerateResponse:
    """Temporary endpoint for testing AI seed generation from one chunk.

    Does not persist seeds or trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await generate_seeds_from_chunk(
        course_id=safe_course_id,
        chunk_id=request.chunk_id,
        count=request.count,
    )

    return SeedGenerateResponse(
        courseId=result["courseId"],
        chunkId=result["chunkId"],
        model=result["model"],
        count=result["count"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
    )


@app.get(
    "/api/starter-generation/status",
    response_model=StarterGenerationStatusResponse,
)
async def starter_generation_status() -> StarterGenerationStatusResponse:
    """Read-only process-local status of the in-flight starter seed job."""
    status = get_starter_job_status()
    return StarterGenerationStatusResponse(
        active=bool(status["active"]),
        courseId=status["courseId"],
        operation=status["operation"],
        startedAt=status["startedAt"],
    )


@app.post(
    "/api/courses/{course_id}/seeds/generate-starter",
    response_model=StarterSeedGenerateResponse,
)
async def generate_course_starter_seeds(
    course_id: str,
    request: StarterSeedGenerateRequest,
) -> StarterSeedGenerateResponse:
    """Temporary endpoint for course-level starter seed generation.

    Set save=true to persist accepted validated seeds.
    Does not trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    operation = "top_up" if request.top_up else "manual"
    async with starter_job_slot(safe_course_id, operation):
        result = await generate_starter_seeds_for_course(
            course_id=safe_course_id,
            target_count=request.target_count,
            save=request.save,
            force_refresh=request.force_refresh,
            top_up=request.top_up,
        )

    # This route persisted seeds and, until now, left starterSeedGeneration
    # describing whichever earlier run wrote it. Reconciliation derives the
    # record from the course's real seed count instead. Best-effort: a status
    # write that fails must not fail a request whose seeds are already saved.
    if request.save:
        await reconcile_starter_seed_generation(
            safe_course_id,
            target_count=result["targetCount"],
        )

    persistence = None
    if result.get("persistence") is not None:
        persistence = StarterSeedPersistence(**result["persistence"])

    return StarterSeedGenerateResponse(
        courseId=result["courseId"],
        model=result["model"],
        targetCount=result["targetCount"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
        progress=StarterSeedProgress(**result["progress"]),
        persistence=persistence,
        localSnapshotPath=public_snapshot_ref(result.get("localSnapshotPath")),
    )


@app.post(
    "/api/courses/{course_id}/seeds/top-up",
    response_model=StarterSeedGenerateResponse,
)
async def top_up_course_starter_seeds(
    course_id: str,
    request: StarterSeedTopUpRequest | None = None,
) -> StarterSeedGenerateResponse:
    """Fill the gap to targetCount without regenerating the course's seeds.

    Reads the course's stored seeds, computes missingCount, generates only that
    many new accepted seeds, dedupes against existing questions, and saves only
    the new ones when save=true (default).
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = request or StarterSeedTopUpRequest()
    async with starter_job_slot(safe_course_id, "top_up"):
        result = await generate_starter_seeds_for_course(
            course_id=safe_course_id,
            target_count=body.target_count,
            save=body.save,
            force_refresh=body.force_refresh,
            top_up=True,
        )

    # The gap this route just closed is exactly what went unrecorded before: a
    # course topped up to its target kept reporting the first run's shortfall.
    if body.save:
        await reconcile_starter_seed_generation(
            safe_course_id,
            target_count=result["targetCount"],
        )

    persistence = None
    if result.get("persistence") is not None:
        persistence = StarterSeedPersistence(**result["persistence"])

    return StarterSeedGenerateResponse(
        courseId=result["courseId"],
        model=result["model"],
        targetCount=result["targetCount"],
        seeds=[GeneratedSeedExample(**seed) for seed in result["seeds"]],
        progress=StarterSeedProgress(**result["progress"]),
        persistence=persistence,
        localSnapshotPath=public_snapshot_ref(result.get("localSnapshotPath")),
    )


@app.post(
    "/api/courses/{course_id}/training-runs",
    response_model=EnqueueTrainingRunResponse,
    status_code=201,
)
def enqueue_course_training_run(
    course_id: str,
    request: EnqueueTrainingRunRequest,
) -> EnqueueTrainingRunResponse:
    """Queue one training run in PostgreSQL.

    One write, one transaction, one store. The run is durable and visible to
    the admin queue the instant this returns 201 — which is the property the
    previous two-store version could not offer: it wrote the queue first and
    mirrored afterwards, so a professor could click Queue training, get a
    success, reload the page that reads PostgreSQL, and find nothing there.

    The duplicate guard is a conditional INSERT inside the same transaction
    (see `db_training_runs.enqueue_training_run`), so two admins clicking at the
    same moment cannot both win: the loser's INSERT matches no rows and it gets
    a 409. That is also what makes a retry safe — a second attempt after a
    timeout is refused rather than queueing the same work twice.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def work(connection):
        if not course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return enqueue_training_run(
            connection,
            safe_course_id,
            mode=request.mode,
            dataset_ref=request.dataset_ref,
            approved_example_count=request.approved_example_count,
            train_examples=request.train_examples,
            validation_examples=request.validation_examples,
        )

    try:
        with translate_db_errors("queueing a training run"):
            with db_connection() as connection:
                created = work(connection)
    except ActiveTrainingRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Same browser-safe view the `/api/db` training-run routes return: no
    # operator account on the claim, no cluster directory in the completion.
    return EnqueueTrainingRunResponse(
        courseId=safe_course_id,
        runId=created["runId"],
        run=public_training_run(created),
    )


@app.post(
    "/api/courses/{course_id}/training-runs/retry",
    response_model=RetryTrainingRunResponse,
    status_code=201,
)
def retry_course_training_run(course_id: str) -> RetryTrainingRunResponse:
    """Retire this course's stale run and queue a replacement for the same data.

    The recovery path for a run the application still believes is active when
    it is not — a job the cluster finished without a completion callback ever
    reaching PostgreSQL. Nothing else can clear that state: the run will not
    finish, and while it is outstanding `canQueueTraining` correctly refuses to
    queue another.

    The previous run is retired, never removed. It stays in the course's
    training history as a terminal run carrying its Slurm job id and its
    reason, `"Superseded by admin retry"`, because an operator asking what this
    course was waiting on must still be able to find the answer.

    The dataset is carried across untouched. No export is rerun, no split is
    recomputed and no approved example is read, so a retry after a fix to the
    training code trains the corrected code on exactly the data the professor
    already approved.

    Everything happens in one transaction, which takes the model request row
    `FOR UPDATE` before it decides anything. A double-clicked button or a
    second admin therefore cannot produce two new runs: the second call waits,
    then sees the freshly queued replacement and is refused with 409.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def work(connection):
        if not course_exists(connection, safe_course_id):
            raise HTTPException(
                status_code=404, detail=f'Course "{safe_course_id}" was not found.'
            )
        return retry_training_run(connection, safe_course_id)

    try:
        with translate_db_errors("retrying a training run"):
            with db_connection() as connection:
                result = work(connection)
    except RetryNotEligibleError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ActiveTrainingRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    created = result["run"]
    superseded = result["superseded"] or {}
    updated_request = result["request"] or {}

    return RetryTrainingRunResponse(
        courseId=safe_course_id,
        runId=created["runId"],
        run=public_training_run(created),
        supersededRunId=superseded.get("runId", ""),
        supersededRun=public_training_run(superseded),
        requestStatus=updated_request.get("status"),
    )


@app.post(
    "/api/courses/{course_id}/facts/inventory",
    response_model=FactInventoryResponse,
)
async def get_course_fact_inventory(
    course_id: str,
    body: FactInventoryRequest | None = None,
) -> FactInventoryResponse:
    """Build or reuse an inspectable global fact inventory for a course syllabus.

    Extraction-only: this DOES NOT generate starter seeds. Shares the same
    per-course cache as starter generation. Pass forceRefresh to rebuild.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or FactInventoryRequest()
    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    inventory = await load_or_build_fact_inventory(
        course_id=safe_course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        force_refresh=request.force_refresh,
    )

    return FactInventoryResponse(
        courseId=safe_course_id,
        model=inventory["model"],
        factCount=inventory["factCount"],
        droppedCount=inventory["droppedCount"],
        duplicatesRemoved=inventory.get("duplicatesRemoved", 0),
        fallbackUsed=inventory["fallbackUsed"],
        cached=bool(inventory.get("cached")),
        countsByScope=inventory["countsByScope"],
        countsByKind=inventory["countsByKind"],
        countsBySeries=inventory.get("countsBySeries", {}),
        facts=[FactInventoryItem(**fact) for fact in inventory["facts"]],
    )


@app.post(
    "/api/courses/{course_id}/facts/allocation",
    response_model=FactAllocationResponse,
)
async def get_course_fact_allocation(
    course_id: str,
    body: FactAllocationRequest | None = None,
) -> FactAllocationResponse:
    """Build/reuse fact inventory and allocate question slots (inspection).

    Allocation-only: this DOES NOT generate starter seeds, does not persist
    seeds, and is not wired into the live starter-generation pipeline.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or FactAllocationRequest()
    target_count = request.target_count

    storage = get_course_artifact_storage()
    index_data = storage.load_index(safe_course_id)
    if index_data is None:
        raise HTTPException(
            status_code=404,
            detail="Course syllabus index was not found.",
        )

    raw_chunks = index_data.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise HTTPException(
            status_code=404,
            detail=f'No syllabus chunks found for course "{safe_course_id}".',
        )

    inventory = await load_or_build_fact_inventory(
        course_id=safe_course_id,
        raw_chunks=raw_chunks,
        storage=storage,
        force_refresh=request.force_refresh,
    )
    allocation = allocate_slots(
        inventory["facts"],
        target_count=target_count,
    )
    summary = allocation["summary"]

    return FactAllocationResponse(
        courseId=safe_course_id,
        model=inventory["model"],
        factCount=inventory["factCount"],
        droppedCount=inventory["droppedCount"],
        duplicatesRemoved=inventory.get("duplicatesRemoved", 0),
        fallbackUsed=inventory["fallbackUsed"],
        facts=[FactInventoryItem(**fact) for fact in inventory["facts"]],
        allocations=[
            FactAllocationItem(**item) for item in allocation["allocations"]
        ],
        summary=FactAllocationSummary(
            targetCount=summary["targetCount"],
            allocatedSlots=summary["allocatedSlots"],
            byScope=summary["byScope"],
            byKind=summary["byKind"],
            bySeries=summary["bySeries"],
            skippedFacts=[
                FactAllocationSkippedFact(**item)
                for item in summary["skippedFacts"]
            ],
            cappedFacts=[
                FactAllocationCappedFact(**item) for item in summary["cappedFacts"]
            ],
            caps=summary["caps"],
            courseWideAllocated=summary["courseWideAllocated"],
            courseWideReserve=summary["courseWideReserve"],
        ),
        ranking=[
            FactAllocationRankingItem(**item) for item in allocation["ranking"]
        ],
    )


def _course_seed_records(course_id: str) -> list[dict]:
    """Every stored seed for one course, newest first.

    One connection, one read, course-scoped by the repository. Replaces the
    node fetch these routes used to do: the shape they consume — a list of
    records each carrying its own `id` — is unchanged, so review, quality
    inspection, and the training export all kept their parsing.
    """
    with translate_db_errors("loading course seeds"):
        with db_connection() as connection:
            return list_seeds(connection, course_id)


@app.get(
    "/api/courses/{course_id}/seeds",
    response_model=CourseSeedListResponse,
)
async def list_course_seeds(course_id: str) -> CourseSeedListResponse:
    """List one course's stored seeds for review."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    seeds = _course_seed_records(safe_course_id)
    return CourseSeedListResponse(
        courseId=safe_course_id,
        count=len(seeds),
        seeds=[SeedReviewRecord(**seed) for seed in seeds],
    )


@app.post(
    "/api/courses/{course_id}/seeds/{seed_id}/review",
    response_model=SeedReviewResponse,
)
async def review_course_seed(
    course_id: str,
    seed_id: str,
    body: SeedReviewRequest,
) -> SeedReviewResponse:
    """Approve, reject, or edit one seed. Edits preserve grounding provenance."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = body.review_status.strip().lower()
    if status not in REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"reviewStatus must be one of {sorted(REVIEW_STATUSES)}.",
        )

    # `review_seed` applies the same `apply_seed_review` rules the `/api/db`
    # route uses, so the two paths cannot drift apart on provenance: the first
    # edit snapshots originalQuestion/originalAnswer, `edited` survives a later
    # approval, and grounding fields are preserved either way.
    def work(connection):
        return review_seed(
            connection,
            safe_course_id,
            seed_id,
            review_status=status,
            question=body.question,
            answer=body.answer,
            review_notes=body.review_notes,
        )

    try:
        with translate_db_errors("reviewing a seed"):
            with db_connection() as connection:
                stored = work(connection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if stored is None:
        raise HTTPException(status_code=404, detail=f'Seed "{seed_id}" was not found.')

    return SeedReviewResponse(
        courseId=safe_course_id,
        seedId=seed_id,
        seed=SeedReviewRecord(**stored),
    )


@app.post(
    "/api/courses/{course_id}/seeds/quality-check",
    response_model=SeedQualityCheckResponse,
)
async def quality_check_course_seeds(
    course_id: str,
    body: SeedQualityCheckRequest | None = None,
) -> SeedQualityCheckResponse:
    """Inspect a course's stored seeds for coverage and quality flags."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request = body or SeedQualityCheckRequest()
    seeds = _course_seed_records(safe_course_id)
    if request.review_statuses:
        allowed = {
            item.strip().lower()
            for item in request.review_statuses
            if isinstance(item, str) and item.strip()
        }
        seeds = [
            seed for seed in seeds if resolve_review_status(seed) in allowed
        ]

    report = inspect_seed_dataset(seeds)
    return SeedQualityCheckResponse(courseId=safe_course_id, report=report)


@app.post(
    "/api/courses/{course_id}/seeds/export-approved",
    response_model=SeedExportApprovedResponse,
)
async def export_approved_course_seeds(
    course_id: str,
    body: SeedExportApprovedRequest | None = None,
) -> SeedExportApprovedResponse:
    """Export approved-only JSONL + metadata under data/exports/{courseId}/."""
    del body  # reserved; approved-only is always enforced
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    seeds = _course_seed_records(safe_course_id)
    try:
        summary = export_approved_seeds(course_id=safe_course_id, seeds=seeds)
    except FinetuneJsonlValidationError as exc:
        raise HTTPException(status_code=422, detail=public_message(exc)) from exc
    # The summary written to disk keeps the absolute paths it always had; only
    # the copy that leaves over HTTP is made repository-relative.
    return SeedExportApprovedResponse(
        courseId=safe_course_id, summary=public_export_summary(summary)
    )


@app.get(
    "/api/courses/{course_id}/seeds/approved-export-status",
    response_model=ApprovedExportStatusResponse,
)
async def get_approved_export_status(course_id: str) -> ApprovedExportStatusResponse:
    """Report whether approved-finetune.jsonl exists for this course."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = public_export_status(approved_export_status(safe_course_id))
    return ApprovedExportStatusResponse(
        courseId=status["courseId"],
        exists=status["exists"],
        exportPath=status["exportPath"],
        exampleCount=status["exampleCount"],
        sourceFile=status["sourceFile"],
    )


@app.get("/api/training/launch-capability", response_model=TrainingLaunchCapabilityResponse)
def training_launch_capability() -> TrainingLaunchCapabilityResponse:
    """Report whether this backend can submit a training job.

    Lets the admin UI show an honest disabled state instead of offering a
    button that cannot work on this host.
    """
    capability = describe_capability()
    return TrainingLaunchCapabilityResponse(
        enabled=capability.enabled, reason=capability.reason
    )


@app.post(
    "/api/courses/{course_id}/training/launch",
    response_model=TrainingLaunchResponse,
)
def launch_course_training(
    course_id: str,
    body: TrainingLaunchRequest | None = None,
) -> TrainingLaunchResponse:
    """Sync one course's prepared dataset and submit its QLoRA job.

    The whole infrastructure boundary lives behind this endpoint: the browser
    never runs ssh, rsync, or sbatch. Everything is scoped to `course_id`, and
    the underlying scripts refuse to touch another course's export or the live
    inference adapter.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mode = (body.mode if body is not None else "full") or "full"

    try:
        result = launch_training(safe_course_id, mode=mode)
    except LaunchDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LaunchValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LaunchExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TrainingLaunchResponse(
        courseId=safe_course_id,
        jobId=result.job_id,
        mode=result.mode,
        submittedAt=result.submitted_at,
        trainCount=result.train_count,
        validationCount=result.validation_count,
    )


@app.post(
    "/api/courses/{course_id}/seeds/prepare-training-split",
    response_model=PrepareTrainingSplitResponse,
)
async def prepare_course_training_split(
    course_id: str,
    body: PrepareTrainingSplitRequest | None = None,
) -> PrepareTrainingSplitResponse:
    """Create deterministic train/validation split from approved-finetune.jsonl."""
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    split_seed = DEFAULT_SPLIT_SEED
    if body is not None and body.split_seed is not None:
        split_seed = int(body.split_seed)

    try:
        summary = prepare_training_split(
            safe_course_id,
            split_seed=split_seed,
        )
    except TrainingSplitError as exc:
        raise HTTPException(status_code=422, detail=public_message(exc)) from exc
    # `manifest.json` on disk is unchanged — the cluster worker verifies against
    # it — and this is the browser's copy of the same summary.
    return PrepareTrainingSplitResponse(
        courseId=safe_course_id, summary=public_export_summary(summary)
    )
