from fastapi import FastAPI

app = FastAPI(title="CSS360 Syllabus Model Backend")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "css360-syllabus-model-backend",
    }
