import os

# Load backend/.env before other app modules read configuration.
from app import config as _config  # noqa: F401
from app.config import load_backend_env

load_backend_env()

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.course_id import assert_valid_course_id
from app.course_index import build_course_rag_index
from app.course_rag import generate_course_rag_answer
from app.ollama import generate_base_model_response
from app.schemas import (
    BaseModelGenerateRequest,
    BaseModelGenerateResponse,
    CourseChunkMetadata,
    CourseChunksResponse,
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
    RagGenerateRequest,
    RagGenerateResponse,
    RagGenerateSource,
    RagRetrieveResult,
    SeedGenerateRequest,
    SeedGenerateResponse,
    StarterSeedGenerateRequest,
    StarterSeedGenerateResponse,
    StarterSeedPersistence,
    StarterSeedProgress,
    SyllabusTextResponse,
    SyllabusUploadResponse,
)
from app.seed_allocation import allocate_slots
from app.seed_generation import generate_seeds_from_chunk, generate_starter_seeds_for_course
from app.fact_inventory_cache import load_or_build_fact_inventory
from app.starter_jobs import (
    run_auto_starter_seed_generation,
    try_queue_starter_seed_generation,
)
from app.storage import get_course_artifact_storage
from app.syllabus_extract import extract_clean_syllabus_text
from app.syllabus_upload import SyllabusUploadError, validate_syllabus_upload

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


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "syllabus-model-lab-backend",
    }


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
        )
        for chunk in index_data.get("chunks", [])
    ]

    return CourseChunksResponse(
        courseId=safe_course_id,
        chunkCount=len(chunks),
        chunks=chunks,
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

    Does not persist seeds to Firebase or trigger course creation.
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


@app.post(
    "/api/courses/{course_id}/seeds/generate-starter",
    response_model=StarterSeedGenerateResponse,
)
async def generate_course_starter_seeds(
    course_id: str,
    request: StarterSeedGenerateRequest,
) -> StarterSeedGenerateResponse:
    """Temporary endpoint for course-level starter seed generation.

    Set save=true to persist accepted validated seeds to Firebase.
    Does not trigger course creation.
    """
    try:
        safe_course_id = assert_valid_course_id(course_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await generate_starter_seeds_for_course(
        course_id=safe_course_id,
        target_count=request.target_count,
        save=request.save,
        force_refresh=request.force_refresh,
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
