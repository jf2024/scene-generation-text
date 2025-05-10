from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import glob
import pandas as pd

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
print(f"API Key loaded (first 8 chars): {api_key[:8]}...")

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
        
    def initialize_vector_store(self, data_dir: str):
        """Initialize the vector store with script examples from the annotations
        
        Args:
            data_dir: Base directory containing manual_annotations
        """
        structured_data = self.parse_data(data_dir)
        
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
            try:
                script_meta = metadata_df[metadata_df['imdbid'] == script_id].iloc[0] if not metadata_df[metadata_df['imdbid'] == script_id].empty else None
            except IndexError:
                print(f"Warning: No metadata found for script ID {script_id}")
                script_meta = None
            
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
            input_variables=["genre", "setting", "idea", "director_style", "length", "examples"],
            template="""You are an award-winning screenwriter with expertise in multiple genres. Write a compelling, professional-quality scene based on these specifications:

                    SPECIFICATIONS:
                    - Genre: {genre}
                    - Setting: {setting}
                    - Core Idea: {idea}
                    - Director Style: {director_style}
                    - Length: {length}

                    STRICT LENGTH REQUIREMENTS:
                    You must adhere to these exact specifications based on the requested length:
                    - "short": 1-2 pages (approx. 250-500 words), 1-2 paragraphs of description, 3-5 dialogue exchanges
                    - "medium": 3-4 pages (approx. 750-1000 words), 3-4 paragraphs of description, 8-12 dialogue exchanges
                    - "long": 5-7 pages (approx. 1250-1750 words), 5+ paragraphs of description, 15+ dialogue exchanges

                    SCREENPLAY FORMAT REQUIREMENTS:
                    1. SCENE HEADING: Use proper INT./EXT. format with LOCATION and TIME OF DAY
                    2. SCENE DESCRIPTION: Write vivid, visual descriptions in present tense. Focus on what can be SEEN and HEARD
                    3. CHARACTER NAMES: Use ALL CAPS when first introducing a character
                    4. DIALOGUE: Format with character name centered above their lines
                    5. PARENTHETICALS: Use sparingly for essential acting cues (beat), (whispers), etc.
                    6. Follow proper spacing between elements: double space between heading/action/dialogue blocks

                    STORYTELLING GUIDANCE:
                    1. Begin in media res - drop viewers into an active moment
                    2. Create tension or conflict within the scene
                    3. Reveal character through action and dialogue, not exposition
                    4. Include sensory details that establish mood and atmosphere
                    5. Pay close attention to pacing - vary sentence length for rhythm
                    6. End with a compelling moment that moves the story forward

                    STUDY THESE REFERENCE EXAMPLES:
                    {examples}

                    DIRECTOR STYLE NOTES:
                    - Analyze the visual language, pacing, and emotional tone in the examples
                    - Incorporate elements that reflect {director_style}'s signature techniques
                    - Consider how {director_style} would frame shots and direct actor performances

                    Now write a compelling scene that authentically reflects the {genre} genre and {director_style}'s cinematic style. Follow the EXACT length specifications for {length} length. Count your dialogue exchanges and descriptions to ensure compliance.
                    """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def write_scene(self, genre: str, setting: str, idea: str, director_style: str, length: str, examples: List[str]) -> str:
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
            input_variables=["scene", "genre", "director_style", "length"],
            template="""You are a renowned script doctor who has edited award-winning screenplays across multiple genres. Critically analyze and elevate the following scene while strictly maintaining the specified length:

                        ORIGINAL SCENE:
                        {scene}

                        SCENE PARAMETERS:
                        - Genre: {genre}
                        - Director Style: {director_style}
                        - Target Length: {length}

                        STRICT LENGTH REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:
                        - "short": 1-2 pages (approx. 250-500 words), 1-2 paragraphs of description, 3-5 dialogue exchanges
                        - "medium": 3-4 pages (approx. 750-1000 words), 3-4 paragraphs of description, 8-12 dialogue exchanges
                        - "long": 5-7 pages (approx. 1250-1750 words), 5+ paragraphs of description, 15+ dialogue exchanges

                        EDITING PROCESS (Complete each step):

                        STEP 1: LENGTH EVALUATION
                        - Count words, paragraphs, and dialogue exchanges in the original scene
                        - Determine if adjustments are needed to meet the length requirements for {length}
                        - Plan to add or remove content as needed while preserving core story elements

                        STEP 2: STRUCTURAL ASSESSMENT
                        - Verify proper screenplay format elements:
                          * Scene headings (INT./EXT. LOCATION - TIME OF DAY)
                          * Action blocks (present tense, visual language)
                          * Character introductions (ALL CAPS on first appearance)
                          * Dialogue formatting (character name centered above lines)
                          * Parentheticals (minimal, only when necessary)
                          * Transitions and spacing (maintain industry-standard format)

                        STEP 3: CONTENT ENHANCEMENT
                        - Strengthen these narrative elements:
                          * Opening hook (create immediate engagement)
                          * Character objectives (ensure each character has clear motivation)
                          * Conflict development (heighten tension appropriately for {genre})
                          * Subtext (reduce on-the-nose dialogue, add layers of meaning)
                          * Scene purpose (ensure scene advances plot or reveals character)
                          * Ending impact (create momentum for next scene)

                        STEP 4: STYLISTIC REFINEMENT
                        - Apply {director_style}'s signature techniques:
                          * Visual language (adjust descriptions to match director's visual style)
                          * Pacing (modify rhythm of action and dialogue to match director's tempo)
                          * Emotional tone (adjust for director's typical emotional register)
                          * Shot suggestions (subtly imply camera work typical of this director)

                        STEP 5: GENRE AUTHENTICITY
                        - Intensify elements that make this authentically {genre}:
                          * Tropes (use or subvert genre conventions purposefully)
                          * Language (adjust dialogue/descriptions to genre expectations)
                          * Atmosphere (enhance sensory details that establish genre feel)

                        STEP 6: FINAL LENGTH VERIFICATION
                        - Count words, paragraphs, and dialogue exchanges in your edited scene
                        - Make final adjustments to ensure it matches the required length for {length}

                        Return only the polished, production-ready scene in perfect screenplay format. Maintain the core story but significantly elevate the craft. The final scene MUST comply with the exact length specifications for {length} length.
                        """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def edit_scene(self, scene: str, genre: str, director_style: str, length: str) -> str:
        """Edit and improve the given scene"""
        return self.chain.run(
            scene=scene,
            genre=genre,
            director_style=director_style,
            length=length
        )

class ScriptGenerationCrew:
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.writer_agent = WriterAgent()
        self.editor_agent = EditorAgent()
        
    def initialize_with_data(self, data_dir: str):
        """Initialize the RAG agent with annotated scripts"""
        self.rag_agent.initialize_vector_store(data_dir)
    
    def generate_scene(self, 
                      genre: str, 
                      setting: str, 
                      idea: str,
                      director_style: str,
                      length: str,
                      debug: bool = False) -> Dict[str, str]:
        """Generate a complete scene using all agents
        
        Args:
            genre: The genre of the scene (e.g., "Science Fiction", "Drama")
            setting: The location or environment for the scene (e.g., "Space Station")
            idea: The core concept or conflict for the scene
            director_style: The director whose style to emulate (e.g., "Steven Spielberg")
            length: Scene length - "short", "medium", or "long"
            debug: Whether to return detailed debug information
            
        Returns:
            Dictionary containing the initial scene, final scene, examples used, and metrics
        """
        # Validate and standardize length parameter
        length = length.lower().strip()
        if length not in ["short", "medium", "long"]:
            length = "medium"
        
        # 1. Retrieve relevant examples
        query = f"{genre} {setting} {idea} {director_style}"
        relevant_examples = self.rag_agent.retrieve_relevant_content(query, k=3)
        
        # 2. Generate initial scene
        initial_scene = self.writer_agent.write_scene(
            genre=genre,
            setting=setting,
            idea=idea,
            director_style=director_style,
            length=length,
            examples=relevant_examples
        )
        
        initial_word_count = len(initial_scene.split())
        initial_dialogue_count = initial_scene.count("\n\n") // 2
        
        # 3. Edit the scene
        final_scene = self.editor_agent.edit_scene(
            scene=initial_scene,
            genre=genre,
            director_style=director_style,
            length=length
        )
        
        # Calculate metrics
        final_word_count = len(final_scene.split())
        final_dialogue_count = final_scene.count("\n\n") // 2
        
        # Return comprehensive results
        result = {
            "initial_scene": initial_scene,
            "final_scene": final_scene,
            "examples_used": relevant_examples if debug else [],
            "metrics": {
                "length_setting": length,
                "initial_word_count": initial_word_count,
                "final_word_count": final_word_count,
                "word_count_change": final_word_count - initial_word_count,
                "initial_dialogue_approx": initial_dialogue_count,
                "final_dialogue_approx": final_dialogue_count
            }
        }
        
        if not debug:
            result.pop("examples_used", None)
            
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the crew
    crew = ScriptGenerationCrew()
    
    # Initialize with annotated data
    crew.initialize_with_data(
        data_dir="data"
    )
    
    # Generate a scene
    result = crew.generate_scene(
        genre="Science Fiction",
        setting="Space Station",
        idea="A crew member discovers an alien artifact that seems to be alive",
        director_style="Steven Spielberg",
        length="long"
    )
    
    print("Final Scene:")
    print(result["final_scene"]) 