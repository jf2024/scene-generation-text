from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import glob
import pandas as pd

load_dotenv()

def parse_annotated_script(file_path):
        """
        Parse an annotated script file into structured format
        Returns a list of dictionaries for each scene in the script
        """
        # Creates structure of output
        scenes = []
        current_scene = {
            "scene_heading": "",
            "description": "",
            "dialog": [],
            "speakers": [],
            "script_id": ""
        }
        script_id = file_path.split('_')[-2]
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # For each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split into label and actual text
            if ': ' in line:
                label, content = line.split(': ', 1)
                # If label is a new scene heading, add previous to scene
                if label == 'scene_heading':
                    if current_scene["scene_heading"]:
                        current_scene["script_id"] = script_id
                        scenes.append(current_scene)
                        current_scene = {
                            "scene_heading": content,
                            "description": "",
                            "dialog": [],
                            "speakers": [],
                            "script_id": script_id
                        }
                    else:
                        current_scene["scene_heading"] = content
                        current_scene["script_id"] = script_id
        
                elif label == 'text':
                    current_scene["description"] += content + " "
                elif label == 'dialog':
                    current_scene["dialog"].append(content)
                elif label == 'speaker_heading':
                    current_scene["speakers"].append(content)
        # Add final scene         
        if current_scene["scene_heading"]:
            current_scene["script_id"] = script_id
            scenes.append(current_scene)
            
        return scenes


class RAGAgent:
    def __init__(self, embeddings_model=None, vector_store=None):
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.vector_store = vector_store

    def parse_data(self, data_dir: str) -> List[Dict]:
        """
        Load script data from the annotations directory  
        Args:
            data_dir: Base directory (data) containing manual_annotations
        """
        structured_data = []
        # Determine which annotations to use
        anno_path = os.path.join(data_dir, 'manual_annotations', 'manual_annotations', '*.txt')    
        # Load all annotation files
        for anno_file in glob.glob(anno_path):
            # extend() adds all elements from that list to the structured_data list
                # A list of dictionaries explaining the scene
            structured_data.extend(parse_annotated_script(anno_file))
        return structured_data
        
    def initialize_vector_store(self, data_dir: str, use_manual: bool = True):
        """Initialize the vector store with script examples from the annotations
        
        Args:
            data_dir: Base directory containing manual_annotations and BERT_annotations
            use_manual: Whether to use manual annotations (True) or BERT annotations (False)
        """
        structured_data = self.load_script_data(data_dir, use_manual)
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'movie_meta_data.csv')
        metadata_df = pd.read_csv(metadata_path)
        # Convert imdbid to string with leading zeros to match file naming
        metadata_df['imdbid'] = metadata_df['imdbid'].astype(str).str.zfill(7)
        
        # Prepare documents for indexing
        documents = []
        
        # Convert structured scenes to searchable text
        for scene in structured_data:
            # Extract script filename to get imdbid
            script_id = scene.get('script_id', '')  # You'll need to add this in parse_annotated_script
            
            # Get metadata for this script
            script_meta = metadata_df[metadata_df['imdbid'] == script_id].iloc[0] if not metadata_df[metadata_df['imdbid'] == script_id].empty else None
            
            # Format dialog with speakers for better context
            dialog_with_speakers = []
            for speaker, line in zip(scene['speakers'], scene['dialog']):
                if speaker and line:
                    dialog_with_speakers.append(f"{speaker}: {line}")
            
            # Build metadata section
            metadata_text = ""
            if script_meta is not None:
                metadata_text = f"""
                Title: {script_meta['title']}
                Genre: {script_meta['genres']}
                Plot: {script_meta['plot']}
                Keywords: {script_meta['keywords']}
                Director: {script_meta['directors']}
                Cast: {script_meta['cast']}
                """
            
            scene_text = f"""
            {metadata_text}
            Scene: {scene['scene_heading']}
            Description: {scene['description']}
            Dialog:
            {chr(10).join(dialog_with_speakers)}
            """
            documents.append(scene_text)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = text_splitter.create_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
    
    def retrieve_relevant_content(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant script examples based on the query"""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

class WriterAgent:
    def __init__(self, temperature: float = 0.7):
        self.llm = ChatOpenAI(temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["genre", "setting", "idea", "examples"],
            template="""You are an experienced screenwriter. Write a compelling scene based on the following criteria:
            Genre: {genre}
            Setting: {setting}
            Core Idea: {idea}
            
            Here are some example scenes for reference:
            {examples}
            
            Write a scene that follows proper screenplay format:
            1. Start with a scene heading (INT/EXT, location, time)
            2. Include clear scene descriptions
            3. Format dialog with speaker names in caps
            4. Add parentheticals for important acting cues
            
            Make the scene original and engaging while maintaining professional formatting.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def write_scene(self, genre: str, setting: str, idea: str, director_style: Optional[str], length: str,examples: List[str]) -> str:
        """Generate a scene based on the given criteria and examples"""
        examples_text = "\n\n".join(examples) if examples else "No examples provided."
        return self.chain.run(
            genre=genre,
            setting=setting,
            idea=idea,
            director_style=director_style,
            length=length,
            examples=examples_text
        )

class EditorAgent:
    def __init__(self, temperature: float = 0.3):
        self.llm = ChatOpenAI(temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["scene", "genre", "feedback_points"],
            template="""You are an experienced script editor. Review and improve the following scene:

            Scene:
            {scene}

            Genre: {genre}
            Consider these specific points:
            {feedback_points}

            Improve the scene while maintaining proper screenplay format:
            1. Scene headings (INT/EXT, location, time)
            2. Action descriptions (present tense, visual)
            3. Character names in caps when first introduced
            4. Dialog formatting and parentheticals
            5. Proper spacing and structure

            Focus on:
            1. Dialogue authenticity
            2. Scene pacing
            3. Character development
            4. Visual storytelling
            5. Genre consistency

            Return the improved scene in proper screenplay format.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def edit_scene(self, scene: str, genre: str, feedback_points: List[str]) -> str:
        """Edit and improve the given scene"""
        feedback_text = "\n".join(feedback_points)
        return self.chain.run(
            scene=scene,
            genre=genre,
            feedback_points=feedback_text
        )

class ScriptGenerationCrew:
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.writer_agent = WriterAgent()
        self.editor_agent = EditorAgent()
        
    def initialize_with_data(self, data_dir: str, use_manual: bool = True):
        """Initialize the RAG agent with annotated scripts"""
        self.rag_agent.initialize_vector_store(data_dir, use_manual)
    
    def generate_scene(self, 
                      genre: str, 
                      setting: str, 
                      idea: str,
                      director_style: str,
                      length: str) -> Dict[str, str]:
        """Generate a complete scene using all agents"""
        
        # 1. Retrieve relevant examples
        relevant_examples = self.rag_agent.retrieve_relevant_content(
            f"{genre} {setting} {idea}"
        )
        
        # 2. Generate initial scene
        initial_scene = self.writer_agent.write_scene(
            genre=genre,
            setting=setting,
            idea=idea,
            director_style=director_style,
            length=length,
            examples=relevant_examples
        )
            
        final_scene = self.editor_agent.edit_scene(
            scene=initial_scene,
            genre=genre
        )
        
        return {
            "initial_scene": initial_scene,
            "final_scene": final_scene,
            "examples_used": relevant_examples
        }

# Example usage
if __name__ == "__main__":
    # Initialize the crew
    crew = ScriptGenerationCrew()
    
    # Initialize with annotated data
    crew.initialize_with_data(
        data_dir="data",
        use_manual=True  # Use manual annotations (more accurate)
    )
    
    # Generate a scene
    result = crew.generate_scene(
        genre="Science Fiction",
        setting="Space Station",
        idea="A crew member discovers an alien artifact that seems to be alive"
    )
    
    print("Final Scene:")
    print(result["final_scene"]) 