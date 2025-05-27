import argparse
import json
from src.rag.project_rag import ProjectRAG
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Project RAG System CLI - Activity Extractor with AWS Bedrock")
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force re-ingestion of documents and rebuild the FAISS index."
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Attempting to initialize ProjectRAG system for Activity Extraction with ChatBedrock...")
    rag_system = ProjectRAG()
    
    logger.info(f"Attempting full setup of the RAG system. Force ingest: {args.force_ingest}")
    if rag_system.full_setup(force_rebuild_index=args.force_ingest):
        logger.info("RAG system setup successful.")
        
        queries = [
            ("Development of digital application for the optimal integration of gender-sensitive Community-based landslide early warning system (CBLEWS) and Household Disaster Preparedness and Response Plan (HDPRP)", "Project Manager"),
            ("Development of digital application for the optimal integration of gender-sensitive Community-based landslide early warning system (CBLEWS) and Household Disaster Preparedness and Response Plan (HDPRP)", "Backend developer")
        ]

        for project_name, position in queries:
            logger.info(f"\n--- Querying ---")
            logger.info(f"Project: {project_name}, Position: {position}")
            result = rag_system.query(project_name, position)
            
            if result and 'response' in result:
                logger.info("\nStructured Response:")
                logger.info(json.dumps(result['response'], indent=2))
            else:
                logger.error("Failed to get an answer for this query or query system not ready.")
            logger.info("--------------------")
    else:
        logger.error("Failed to set up the RAG system. Please check the logs above for errors.")
        logger.error("Ensure your 'projects' directory exists and contains valid JSON files.")
        logger.error("Also ensure AWS credentials are properly set in the environment or .env file.")

if __name__ == "__main__":
    main()