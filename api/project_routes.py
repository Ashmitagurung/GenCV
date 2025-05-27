from fastapi import HTTPException, APIRouter
import asyncio
import logging
from typing import Dict, Union # Added Union
from pydantic import BaseModel


from src.rag.project_rag import ProjectRAG
from src.rag.role_rag import RoleRAG 

# Create router
router = APIRouter()

# Dictionary to store RAG instances, keyed by index_id
# Updated to hold either ProjectRAG or RoleRAG instances
rag_instances: Dict[str, Union[ProjectRAG, RoleRAG]] = {}
setup_locks: Dict[str, asyncio.Lock] = {}

class QueryRequest(BaseModel):
    project_name: str
    position: str
    
class RoleRequest(BaseModel):
    position: str

class QueryResponse(BaseModel):
    """Response model for query results."""
    Name_of_the_assignment_or_project: str
    Year: str
    Location: str
    Client: str
    Position: str
    Project_description: str
    Activities_performed: str
    

class RoleResponse(BaseModel):
    """Response model for query results."""
    Position: str
    Activities_performed: str


async def initialize_rag(index_id: str) -> bool:
    """Initialize a RAG instance (ProjectRAG or RoleRAG) for the given index_id."""
    global rag_instances, setup_locks
    
    if index_id in rag_instances:
        logging.info(f"RAG instance '{index_id}' already initialized.")
        return True

    # Ensure a lock exists for this index_id's setup process
    # This lock is primarily used by ensure_rag_ready for post-instantiation setup,
    # but creating it here ensures it's available if initialize_rag is called on-demand.
    if index_id not in setup_locks:
        setup_locks[index_id] = asyncio.Lock()
    else:
        logging.warning(
            f"Setup lock for '{index_id}' existed, but RAG instance was not found. "
            "Proceeding with initialization using existing lock."
        )
    
    try:
        logging.info(f"Initializing RAG instance '{index_id}'...")
        
        rag_instance: Union[ProjectRAG, RoleRAG] # Type hint for the instance
        
        if index_id == "project_rag":
            logging.info(f"Creating ProjectRAG instance for '{index_id}'.")
            rag_instance = ProjectRAG()
        elif index_id == "role_rag":
            logging.info(f"Creating RoleRAG instance for '{index_id}'.")
            rag_instance = RoleRAG()
        else:
            logging.error(f"Unknown RAG type for index_id: {index_id}")
            # Clean up the lock if it was newly created for this failed attempt.
            # For simplicity, we'll leave it; subsequent valid attempts might use it.
            return False
        
        rag_instance.index_id = index_id  # Ensure index_id is set on the RAG instance
        
        # Perform full setup (LLM, documents, vector store, QA chain)
        success = await rag_instance.full_setup(force_rebuild_index=False)
        if not success:
            logging.error(f"Failed to set up RAG instance '{index_id}'.")
            return False
        
        # Store the RAG instance
        rag_instances[index_id] = rag_instance
        # The setup_locks[index_id] is already prepared.
        
        logging.info(f"RAG instance '{index_id}' initialized and set up successfully.")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize RAG instance '{index_id}': {e}", exc_info=True)
        return False

async def cleanup_rag():
    """Cleanup all RAG instances."""
    global rag_instances, setup_locks
    
    try:
        logging.info("Cleaning up all RAG instances...")
        for index_id, rag in rag_instances.items():
            await rag.cleanup() 
            logging.info(f"RAG instance '{index_id}' cleaned up.")
        
        rag_instances.clear()
        setup_locks.clear()
        logging.info("All RAG instances cleaned up successfully.")
    except Exception as e:
        logging.error(f"Error during RAG cleanup: {e}", exc_info=True)

async def ensure_rag_ready(index_id: str) -> Union[ProjectRAG, RoleRAG]:
    """Ensure the RAG instance for the given index_id is ready for queries."""
    if index_id not in rag_instances:
        # This implies the RAG was not initialized at startup or was cleaned up.
        # Attempt to initialize it on-demand.
        logging.warning(f"RAG instance '{index_id}' not found. Attempting dynamic initialization.")
        success = await initialize_rag(index_id=index_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"RAG instance '{index_id}' failed to initialize dynamically.")
    
    rag = rag_instances[index_id]
    
    # Ensure the lock exists (should have been created by initialize_rag)
    if index_id not in setup_locks:
        logging.error(f"CRITICAL: Lock for RAG instance '{index_id}' is missing after initialization attempt!")
        # As a fallback, create a lock, but this indicates a deeper issue.
        setup_locks[index_id] = asyncio.Lock()

    _lock = setup_locks[index_id]
    
    # Check if RAG instance is ready, and set up if not, under lock
    if not rag.is_ready(): # Assuming is_ready() method exists
        async with _lock:
            if not rag.is_ready(): # Double-check inside lock
                logging.info(f"Setting up RAG instance '{index_id}' components (ensure_rag_ready)...")
                success = await rag.full_setup(force_rebuild_index=False) # Assuming full_setup method
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to set up RAG instance '{index_id}'")
                logging.info(f"RAG instance '{index_id}' setup completed and ready for queries.")
    
    return rag

@router.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    instance_info = {}
    for index_id, rag in rag_instances.items():
        if hasattr(rag, 'get_model_info'):
            instance_info[index_id] = rag.get_model_info()
        else:
            instance_info[index_id] = {"status": "info_not_available", "ready": rag.is_ready() if hasattr(rag, 'is_ready') else "unknown"}
            
    return {
        "message": "RAG API is running", # Generalized message
        "status": "healthy",
        "rag_instances": instance_info
    }

@router.post("/query", response_model=QueryResponse)
async def query_project_details(data: QueryRequest): # Renamed function for clarity
    """Endpoint to query project details and activities using ProjectRAG."""
    index_id = "project_rag" # Specific to ProjectRAG
    try:
        project_name = data.project_name.strip()
        position = data.position.strip()
        
        if not project_name or not position:
            raise HTTPException(
                status_code=400, 
                detail="project_name and position are required and cannot be empty"
            )
        
        rag = await ensure_rag_ready(index_id=index_id)
        
        result = await rag.query(project_name, position) # Assuming query method exists
        
        if not result or 'response' not in result:
            raise HTTPException(
                status_code=500, 
                detail="Query failed - no response generated"
            )
        
        return result['response']
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in query_project_details for index_id '{index_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/query-role", response_model=RoleResponse)
async def query_role_activities(data: RoleRequest): # New function name
    """Endpoint to query role-specific details and activities using RoleRAG."""
    index_id = "role_rag" # Specific to RoleRAG
    try:
        # RoleRAG might primarily use 'position'. 'project_name' might be context or ignored.
        position = data.position.strip()
        
        # Adjust validation if RoleRAG has different requirements (e.g., only position is mandatory)
        if not position: # Example: If only position is strictly needed for RoleRAG
             raise HTTPException(
                status_code=400, 
                detail="position is required and cannot be empty for role query"
            )

        rag = await ensure_rag_ready(index_id=index_id)
        

        result = await rag.query(position) 
        
        if not result or 'response' not in result:
            raise HTTPException(
                status_code=500, 
                detail="Query failed - no response generated"
            )
        
        return result['response']
        
    except HTTPException:
        raise
    except Exception as e:
        # Corrected logging to use the local index_id variable
        logging.error(f"Unexpected error in query_role_activities for index_id '{index_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )