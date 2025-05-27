import os
import asyncio
import traceback
import time
import threading
import sys
import warnings
import logging
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import setup_logger
from huggingface_hub import snapshot_download
from tqdm import tqdm

class VectorStore:
    """Manages embedding model initialization and FAISS vector store creation, loading, and saving."""
    
    def __init__(self, model_name: str, model_cache_dir: str, index_path: str, index_id: str):
        self.logger = setup_logger()
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.index_id = index_id
        
        # Force stdout/stderr to be unbuffered for Docker
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        
        # Suppress repetitive HuggingFace warnings
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*Xet Storage.*")
        
        # Enforce FAISS index naming convention: faiss_index_{index_id}
        expected_index_name = f"faiss_index_{index_id}"
        if os.path.basename(index_path) != expected_index_name:
            self.logger.warning(
                f"Index path '{index_path}' does not match expected name 'faiss_index_{index_id}'. "
                f"Adjusting to use '{expected_index_name}'."
            )
            index_path = os.path.join(os.path.dirname(index_path), expected_index_name)
        self.index_path = index_path
        self.embedding_model = None
        self.vectorstore = None
        
        # Ensure directories exist
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def _docker_safe_print(self, message: str, end: str = "\n", flush: bool = True):
        """Docker-safe print function that forces output and logging."""
        print(message, end=end, flush=flush)
        self.logger.info(message.strip())
        sys.stdout.flush()
        sys.stderr.flush()
    
    def _show_informative_progress(self, stop_event: threading.Event, model_name: str, progress_data: dict):
        """Shows informative progress with actual download statistics."""
        start_time = time.time()
        last_update = 0
        
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            if elapsed - last_update >= 5.0:
                if 'files_downloaded' in progress_data and 'total_files' in progress_data:
                    files_progress = f"{progress_data['files_downloaded']}/{progress_data['total_files']} files"
                    if progress_data['total_files'] > 0:
                        pct = (progress_data['files_downloaded'] / progress_data['total_files']) * 100
                        files_progress += f" ({pct:.1f}%)"
                else:
                    files_progress = "Fetching model files..."
                
                size_info = ""
                if 'total_size_mb' in progress_data and progress_data['total_size_mb'] > 0:
                    size_info = f" | {progress_data['total_size_mb']:.1f}MB"
                
                self._docker_safe_print(f"📥 Downloading '{model_name}': {files_progress}{size_info} | Time: {time_str}")
                last_update = elapsed
            
            time.sleep(1.0)
    
    def _get_model_cache_path(self) -> str:
        """Get the cache path for the model."""
        repo_id = self.model_name.replace('/', '--')
        return os.path.join(self.model_cache_dir, f"models--{repo_id}")
    
    def _is_model_cached(self) -> bool:
        """Check if model is already cached."""
        model_cache_path = self._get_model_cache_path()
        snapshots_path = os.path.join(model_cache_path, "snapshots")
        if not os.path.exists(snapshots_path):
            self._docker_safe_print(f"Cache check: Directory {snapshots_path} does not exist")
            return False
        if not os.listdir(snapshots_path):
            self._docker_safe_print(f"Cache check: Directory {snapshots_path} exists but is empty")
            return False
        self._docker_safe_print(f"Cache check: Found model in {snapshots_path}")
        return True
    
    def _load_cached_model(self) -> Optional[HuggingFaceEmbeddings]:
        """Load model from cache with progress indication."""
        model_cache_path = self._get_model_cache_path()
        
        self._docker_safe_print(f"📁 Found cached model for '{self.model_name}' at: {model_cache_path}")
        self._docker_safe_print(f"🔄 Loading embedding model from cache for index_id: {self.index_id}...")
        
        start_time = time.time()
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=self.model_cache_dir,
                model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                encode_kwargs={'normalize_embeddings': True}
            )
            load_time = time.time() - start_time
            self._docker_safe_print(f"✅ Successfully loaded cached embedding model in {load_time:.2f} seconds")
            return embedding_model
        except Exception as e:
            self._docker_safe_print(f"❌ Failed to load cached model: {e}")
            self._docker_safe_print("🔄 Will download fresh model from Hugging Face...")
            return None
    
    def _download_fresh_model(self) -> Optional[HuggingFaceEmbeddings]:
        """Download model from Hugging Face with informative progress indication using tqdm."""
        self._docker_safe_print(f"🌐 No cached model found. Downloading '{self.model_name}' from Hugging Face...")
        self._docker_safe_print(f"📥 Model will be cached at: {self._get_model_cache_path()}")
        
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
        os.environ['TRANSFORMERS_CACHE'] = self.model_cache_dir
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        original_filter = warnings.filters.copy()
        warnings.filterwarnings("ignore", message=".*Xet Storage.*")
        
        start_time = time.time()
        try:
            progress_bar = tqdm(
                desc=f"Downloading '{self.model_name}'",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt} files [{elapsed}<{remaining}]"
            )
            
            self._docker_safe_print(f"📡 Initializing download for '{self.model_name}'...")
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.model_cache_dir,
                local_files_only=False,
                tqdm_class=None
            )
            
            model_cache_path = self._get_model_cache_path()
            if os.path.exists(model_cache_path):
                file_count = len([f for f in os.listdir(model_cache_path) if os.path.isfile(os.path.join(model_cache_path, f))])
                progress_bar.total = file_count
                progress_bar.update(file_count)
            
            progress_bar.close()
            self._docker_safe_print(f"📦 Model files downloaded successfully to: {model_path}")
            self._docker_safe_print(f"🔧 Initializing embedding model...")
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=self.model_cache_dir,
                model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            download_time = time.time() - start_time
            self._docker_safe_print(f"✅ Successfully downloaded and loaded '{self.model_name}' in {download_time:.2f} seconds")
            self._docker_safe_print(f"💾 Model cached for future use at: {self._get_model_cache_path()}")
            
            warnings.filters = original_filter
            return embedding_model
        
        except Exception as e:
            progress_bar.close()
            self._docker_safe_print(f"❌ Model download failed: {e}")
            warnings.filters = original_filter
            return None
    
    def _download_fallback_model(self) -> Optional[HuggingFaceEmbeddings]:
        """Download fallback model with informative progress indication."""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        self._docker_safe_print(f"🔄 Falling back to smaller model: {fallback_model}")
        
        progress_data = {
            'files_downloaded': 0,
            'total_files': 0,
            'total_size_mb': 0.0
        }
        
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=self._show_informative_progress, args=(stop_event, fallback_model, progress_data))
        progress_thread.daemon = True
        progress_thread.start()
        
        start_time = time.time()
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=fallback_model,
                cache_folder=self.model_cache_dir,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            stop_event.set()
            progress_thread.join(timeout=1)
            
            download_time = time.time() - start_time
            self._docker_safe_print(f"✅ Successfully loaded fallback model '{fallback_model}' in {download_time:.2f} seconds")
            return embedding_model
            
        except Exception as e:
            stop_event.set()
            progress_thread.join(timeout=1)
            self._docker_safe_print(f"❌ Fallback model download failed: {e}")
            return None
    
    def initialize_embeddings(self) -> bool:
        """Initializes the HuggingFace embedding model with detailed progress logging."""
        if self.embedding_model is not None:
            self._docker_safe_print(f"Embedding model '{self.model_name}' already initialized for index_id: '{self.index_id}'")
            return True
            
        self._docker_safe_print(f"🚀 Initializing embedding model: '{self.model_name}' for index_id: '{self.index_id}'")
        self._docker_safe_print(f"📂 Model cache directory: {self.model_cache_dir}")
        
        try:
            if self._is_model_cached():
                self.embedding_model = self._load_cached_model()
                if self.embedding_model:
                    return True
            
            self.embedding_model = self._download_fresh_model()
            if self.embedding_model:
                return True
            
            self._docker_safe_print("🔄 Attempting fallback model download...")
            self.embedding_model = self._download_fallback_model()
            if self.embedding_model:
                return True
            
            self._docker_safe_print(f"❌ All embedding model initialization attempts failed for index_id '{self.index_id}'")
            return False
                
        except Exception as e:
            self._docker_safe_print(f"❌ Critical error initializing embedding model for index_id '{self.index_id}': {e}")
            traceback.print_exc()
            return False
    
    async def load_or_create(self, documents: list, force_rebuild: bool = False) -> bool:
        """Loads an existing FAISS index or creates a new one from documents with detailed progress."""
        if self.vectorstore is not None and not force_rebuild:
            self._docker_safe_print(f"Vector store for index_id '{self.index_id}' already loaded")
            return True
            
        if not documents:
            self._docker_safe_print(f"❌ No documents provided for index_id '{self.index_id}'. Cannot setup vector store.")
            return False
        
        self._docker_safe_print(f"🔧 Setting up vector store for index_id: '{self.index_id}'")
        self._docker_safe_print(f"📄 Processing {len(documents)} documents")
        self._docker_safe_print(f"📍 FAISS index path: {self.index_path}")
        
        if not self.embedding_model:
            self._docker_safe_print("🔄 Embedding model not initialized. Initializing now...")
            if not self.initialize_embeddings():
                self._docker_safe_print(f"❌ Failed to initialize embedding model for index_id '{self.index_id}'. Cannot setup vector store.")
                return False
        
        if os.path.exists(self.index_path) and not force_rebuild:
            try:
                self._docker_safe_print(f"📁 Found existing FAISS index at: {self.index_path}")
                self._docker_safe_print(f"🔄 Loading FAISS index for index_id: '{self.index_id}'...")
                
                start_time = time.time()
                self.vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                load_time = time.time() - start_time
                
                self._docker_safe_print(f"✅ Successfully loaded FAISS index in {load_time:.2f} seconds")
                return True
            except Exception as e:
                self._docker_safe_print(f"❌ Error loading FAISS index for index_id '{self.index_id}': {e}")
                self._docker_safe_print("🔄 Will attempt to rebuild index from documents...")
        
        try:
            if force_rebuild:
                self._docker_safe_print(f"🔨 Force rebuild requested. Creating new FAISS index...")
            else:
                self._docker_safe_print(f"🔨 Creating new FAISS index from {len(documents)} documents...")
            
            self._docker_safe_print(f"⏳ Building vector embeddings for index_id: '{self.index_id}'...")
            start_time = time.time()
            
            self.vectorstore = await asyncio.to_thread(
                FAISS.from_documents,
                documents,
                self.embedding_model
            )
            build_time = time.time() - start_time
            self._docker_safe_print(f"✅ Created FAISS vectorstore in {build_time:.2f} seconds")
            
            self._docker_safe_print(f"💾 Saving FAISS index to: {self.index_path}")
            save_start = time.time()
            await asyncio.to_thread(
                self.vectorstore.save_local,
                self.index_path
            )
            save_time = time.time() - save_start
            
            total_time = time.time() - start_time
            self._docker_safe_print(f"✅ Successfully saved FAISS index in {save_time:.2f} seconds")
            self._docker_safe_print(f"🎉 Vector store setup completed in {total_time:.2f} seconds total")
            return True
            
        except Exception as e:
            self._docker_safe_print(f"❌ Error building or saving FAISS index for index_id '{self.index_id}': {e}")
            traceback.print_exc()
            return False
    
    def get_embedding_model(self):
        """Returns the initialized embedding model."""
        return self.embedding_model
    
    def get_vectorstore(self):
        """Returns the loaded or created vectorstore."""
        return self.vectorstore