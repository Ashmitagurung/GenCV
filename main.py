from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.project_routes import router,initialize_rag, cleanup_rag 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    # Startup
    print("Starting up application..hahahahha.")
    
    # Initialize ProjectRAG with default settings (naxa_projects_old_summaries.json)
    success = await initialize_rag(index_id="project_rag")
    if not success:
        print("Error: Failed to initialize ProjectRAG")
        raise Exception("ProjectRAG initialization failed")
    print("ProjectRAG initialized successfully")
    
    success = await initialize_rag(index_id="role_rag")
    if not success:
        print("Error: Failed to initialize OtherRAG")
        raise Exception("RoleRAG initialization failed")
    print("RoleRAG initialized successfully")
    
    yield
    
    # Shutdown
    print("Shutting down application...")
    await cleanup_rag()
    print("ProjectRAG cleanup complete")
    # Uncomment if using OtherRAG
    # await cleanup_other_rag()
    # print("OtherRAG cleanup complete")
    print("Application shutdown complete")

def get_application() -> FastAPI:
    """Get the FastAPI app instance."""
    _app = FastAPI(
        description="GenCV",
        docs_url="/docs",
        redoc_url="/redoc",
        root_path="/project",
        lifespan=lifespan
    )

    # Restrict CORS origins for production (update with your frontend URL)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    _app.include_router(router)
    # Uncomment to include OtherRAG router
    # _app.include_router(other_router)

    return _app

app = get_application()