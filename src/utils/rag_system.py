import os
import json
import numpy as np
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

class SportsVideoRAG:
    """
    Retrieval-Augmented Generation system for sports video analysis.
    This system creates and maintains a vector database of video analysis data,
    which can be queried using natural language via OpenAI LLMs.
    """
    
    def __init__(self, data_dir="data_store", 
                 db_dir="vector_db", 
                 openai_api_key=None):
        """
        Initialize the RAG system.
        
        Args:
            data_dir (str): Directory containing extracted video data
            db_dir (str): Directory for the vector database
            openai_api_key (str): OpenAI API key for embeddings and LLM
        """
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        
        # Set OpenAI API key from parameter or environment variable
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        # Initialize the embedding function
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Create or get collections
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-ada-002"
        )
        
        self.collections = {
            "video_metadata": self.client.get_or_create_collection(
                name="video_metadata", 
                embedding_function=openai_ef
            ),
            "frame_data": self.client.get_or_create_collection(
                name="frame_data", 
                embedding_function=openai_ef
            ),
            "pose_analysis": self.client.get_or_create_collection(
                name="pose_analysis", 
                embedding_function=openai_ef
            )
        }
        
        # Initialize the text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Initialize the OpenAI LLM
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    def load_and_index_video_data(self, reindex=False):
        """
        Load extracted video data from JSON files and index it in the vector database.
        
        Args:
            reindex (bool): Whether to reindex existing data
            
        Returns:
            int: Number of documents indexed
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        # Check for the catalog file
        catalog_file = self.data_dir / "video_catalog.json"
        if not catalog_file.exists():
            raise FileNotFoundError(f"Video catalog file not found at {catalog_file}")
        
        # Load the catalog
        with open(catalog_file, 'r') as f:
            catalog = json.load(f)
        
        # Clear collections if reindexing
        if reindex:
            for collection in self.collections.values():
                collection.delete(where={})
        
        # Process each video in the catalog
        indexed_count = 0
        for video_meta in catalog.get("videos", []):
            # Load the video data file
            video_name = video_meta.get("filename", "").replace(".mp4", "")
            data_file = self.data_dir / f"{video_name}_data.json"
            
            if not data_file.exists():
                print(f"Warning: Data file for {video_name} not found")
                continue
            
            with open(data_file, 'r') as f:
                video_data = json.load(f)
            
            # Index the video metadata
            metadata_text = self._format_metadata_for_indexing(video_data["metadata"])
            self.collections["video_metadata"].add(
                documents=[metadata_text],
                metadatas=[{"video_name": video_name, "type": "metadata"}],
                ids=[f"meta_{video_name}"]
            )
            indexed_count += 1
            
            # Index frame data in chunks
            frame_chunks = self._chunk_frame_data(video_data["frames"], video_name)
            for i, chunk in enumerate(frame_chunks):
                chunk_text = json.dumps(chunk)
                self.collections["frame_data"].add(
                    documents=[chunk_text],
                    metadatas=[{
                        "video_name": video_name, 
                        "type": "frame_data",
                        "chunk_index": i,
                        "frames": f"{chunk[0]['frame_idx']}-{chunk[-1]['frame_idx']}"
                    }],
                    ids=[f"frames_{video_name}_{i}"]
                )
                indexed_count += 1
            
            # Generate and index pose analysis
            pose_analysis = self._generate_pose_analysis(video_data)
            self.collections["pose_analysis"].add(
                documents=[pose_analysis],
                metadatas=[{"video_name": video_name, "type": "analysis"}],
                ids=[f"analysis_{video_name}"]
            )
            indexed_count += 1
            
            print(f"Indexed data for {video_name}")
        
        print(f"Indexed {indexed_count} documents in total")
        return indexed_count
    
    def _format_metadata_for_indexing(self, metadata):
        """Format video metadata as text for indexing."""
        return f"""
        Video: {metadata.get('original_name', '')}
        Dimensions: {metadata.get('width', 0)}x{metadata.get('height', 0)}
        FPS: {metadata.get('fps', 0)}
        Total Frames: {metadata.get('total_frames', 0)}
        Processed Date: {metadata.get('processed_date', '')}
        """
    
    def _chunk_frame_data(self, frames, video_name, chunk_size=50):
        """Split frame data into chunks for indexing."""
        return [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]
    
    def _generate_pose_analysis(self, video_data):
        """Generate a textual analysis of the pose data in the video."""
        # This would ideally use the LLM to generate a detailed analysis
        # For now, we'll create a simple summary
        
        metadata = video_data["metadata"]
        frames = video_data["frames"]
        
        # Sample a few frames to analyze
        sample_indices = np.linspace(0, len(frames)-1, min(5, len(frames)), dtype=int)
        sample_frames = [frames[i] for i in sample_indices]
        
        # Create a textual summary
        analysis = f"""
        Analysis of {metadata.get('original_name', '')}:
        
        This video shows a sports performance with {len(frames)} key frames analyzed.
        The video is {metadata.get('width', 0)}x{metadata.get('height', 0)} at {metadata.get('fps', 0)} FPS.
        
        Key observations:
        - The video contains {metadata.get('total_frames', 0)} total frames
        - The athlete's movements are captured across {len(frames)} sampled frames
        """
        
        # Add some frame-specific information
        if sample_frames:
            analysis += "\n\nFrame samples:\n"
            for frame in sample_frames:
                analysis += f"\nFrame {frame['frame_idx']} (timestamp {frame['timestamp']:.2f}s)"
                if frame.get("metrics"):
                    analysis += f"\nMetrics detected: {list(frame['metrics'].keys())}"
        
        return analysis
    
    def query(self, query_text, k=5):
        """
        Query the RAG system with natural language.
        
        Args:
            query_text (str): Natural language query
            k (int): Number of results to retrieve
            
        Returns:
            str: Response from the LLM
        """
        # Search across all collections
        results = {}
        for name, collection in self.collections.items():
            results[name] = collection.query(
                query_texts=[query_text],
                n_results=k
            )
        
        # Prepare context for the LLM
        context = self._prepare_context_from_results(results, query_text)
        
        # Define the prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are an expert sports analyst AI assistant that analyzes video data.
            Answer the following question about sports video analysis based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a detailed, informative answer that directly addresses the question.
            If you don't know or the information isn't in the context, say so clearly.
            """
        )
        
        # Create and run the chain
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": query_text})
        
        return response
    
    def _prepare_context_from_results(self, results, query):
        """Prepare context from search results for the LLM."""
        context_parts = []
        
        # Add metadata results
        if results["video_metadata"]["documents"]:
            context_parts.append("## Video Information")
            for doc in results["video_metadata"]["documents"][0]:
                context_parts.append(doc)
        
        # Add pose analysis results
        if results["pose_analysis"]["documents"]:
            context_parts.append("\n## Pose Analysis")
            for doc in results["pose_analysis"]["documents"][0]:
                context_parts.append(doc)
        
        # Add frame data results
        if results["frame_data"]["documents"]:
            context_parts.append("\n## Frame Data")
            for doc in results["frame_data"]["documents"][0]:
                # Parse the JSON string back to a dict
                try:
                    frames = json.loads(doc)
                    # Extract a few key frames
                    for i, frame in enumerate(frames[:3]):
                        context_parts.append(f"\nFrame {frame['frame_idx']} (time: {frame['timestamp']:.2f}s):")
                        if frame.get("metrics"):
                            context_parts.append(f"Metrics: {frame['metrics']}")
                except Exception as e:
                    context_parts.append(f"Error parsing frame data: {e}")
        
        return "\n".join(context_parts)
    
    def add_new_video(self, video_path, sample_rate=5):
        """
        Add a new video to the RAG system.
        
        Args:
            video_path (str): Path to the processed video
            sample_rate (int): Sample 1 frame every N frames
            
        Returns:
            bool: Whether the video was successfully added
        """
        # Import the data extractor
        from data_extractor import VideoDataExtractor
        
        # Create extractor and extract video data
        extractor = VideoDataExtractor(output_dir=str(self.data_dir))
        video_data = extractor.extract_video_data(video_path, sample_rate)
        
        # Update the catalog
        catalog_file = self.data_dir / "video_catalog.json"
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
        else:
            catalog = {"videos": [], "total_videos": 0}
        
        # Add the new video to the catalog if not already present
        video_name = video_data["metadata"]["filename"].replace(".mp4", "")
        if not any(v.get("filename") == video_data["metadata"]["filename"] for v in catalog.get("videos", [])):
            catalog["videos"].append(video_data["metadata"])
            catalog["total_videos"] = len(catalog["videos"])
            catalog["updated_at"] = video_data["metadata"]["processed_date"]
            
            with open(catalog_file, 'w') as f:
                json.dump(catalog, f, indent=2)
        
        # Index the new video data
        # Index the video metadata
        metadata_text = self._format_metadata_for_indexing(video_data["metadata"])
        self.collections["video_metadata"].add(
            documents=[metadata_text],
            metadatas=[{"video_name": video_name, "type": "metadata"}],
            ids=[f"meta_{video_name}"]
        )
        
        # Index frame data in chunks
        frame_chunks = self._chunk_frame_data(video_data["frames"], video_name)
        for i, chunk in enumerate(frame_chunks):
            chunk_text = json.dumps(chunk)
            self.collections["frame_data"].add(
                documents=[chunk_text],
                metadatas=[{
                    "video_name": video_name, 
                    "type": "frame_data",
                    "chunk_index": i,
                    "frames": f"{chunk[0]['frame_idx']}-{chunk[-1]['frame_idx']}"
                }],
                ids=[f"frames_{video_name}_{i}"]
            )
        
        # Generate and index pose analysis
        pose_analysis = self._generate_pose_analysis(video_data)
        self.collections["pose_analysis"].add(
            documents=[pose_analysis],
            metadatas=[{"video_name": video_name, "type": "analysis"}],
            ids=[f"analysis_{video_name}"]
        )
        
        print(f"Added and indexed new video: {video_name}")
        return True


if __name__ == "__main__":
    # Simple test - requires OPENAI_API_KEY in environment
    rag = SportsVideoRAG()
    # First index data from extracted videos
    rag.load_and_index_video_data()
    # Then try a query
    response = rag.query("What sports are shown in the videos?")
    print(response) 