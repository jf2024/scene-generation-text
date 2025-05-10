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
import re

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

api_key = os.getenv("OPENAI_API_KEY")

LENGTH_REQUIREMENTS = {
    "short": {
        "min_words": 180,
        "max_words": 250,  
        "min_paragraphs": 1,
        "max_paragraphs": 2,
        "min_exchanges": 3,
        "max_exchanges": 5,
        "description": "Brief moment (200-250 words)",
        "display": "SHORT"
    },
    "medium": {
        "min_words": 400,  
        "max_words": 550,  
        "min_paragraphs": 3,
        "max_paragraphs": 5, 
        "min_exchanges": 6,
        "max_exchanges": 10, 
        "description": "Complete scene (400-550 words)",
        "display": "MEDIUM"
    },
    "long": {
        "min_words": 800,
        "max_words": 1200,
        "min_paragraphs": 5,
        "max_paragraphs": 8,  
        "min_exchanges": 12,
        "max_exchanges": 18, 
        "description": "Detailed scene (800-1200 words)",
        "display": "LONG"
    },
    "extra_long": {
        "min_words": 1500,
        "max_words": 2200, 
        "min_paragraphs": 8,
        "max_paragraphs": 14, 
        "min_exchanges": 18,
        "max_exchanges": 28, 
        "description": "Extended dramatic scene (1500-2200 words)",
        "display": "EXTRA LONG"
    }
}

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
        self.length_requirements = LENGTH_REQUIREMENTS
        
        self.prompt = PromptTemplate(
            input_variables=["genre", "setting", "idea", "director_style", "length", 
                           "examples", "expected_word_count", "expected_paragraphs", 
                           "expected_exchanges", "length_description"],
            template="""You are an award-winning screenwriter with expertise in multiple genres. Write a compelling, professional-quality scene based on these specifications:

                    SPECIFICATIONS:
                    - Genre: {genre}
                    - Setting: {setting}
                    - Core Idea: {idea}
                    - Director Style: {director_style}
                    - Length: {length} ({length_description})

                    STRICT LENGTH REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:
                    - Word count: MINIMUM {expected_word_count} words - AIM FOR THE UPPER RANGE!
                    - Action paragraphs: {expected_paragraphs}
                    - Dialogue exchanges: {expected_exchanges}
                    
                    KEY: For "medium" length or longer, make sure the scene is SUBSTANTIAL with detailed descriptions and rich dialogue.

                    SCREENPLAY FORMAT REQUIREMENTS:
                    1. SCENE HEADING: Use proper INT./EXT. format with LOCATION and TIME OF DAY
                    2. SCENE DESCRIPTION: Write vivid, visual descriptions in present tense with rich sensory details
                    3. CHARACTER NAMES: Use ALL CAPS when first introducing a character
                    4. DIALOGUE: Format with character name centered above their lines
                    5. PARENTHETICALS: Use sparingly for essential acting cues

                    STORYTELLING GUIDANCE:
                    1. Begin in media res - drop viewers into an active moment
                    2. Create tension or conflict within the scene
                    3. Reveal character through action and dialogue
                    4. Include sensory details that establish mood
                    5. End with a compelling moment that moves story forward
                    6. For longer scenes, include more character development and emotional beats

                    STUDY THESE REFERENCE EXAMPLES:
                    {examples}

                    DIRECTOR STYLE NOTES:
                    - Analyze the visual language in the examples
                    - Incorporate elements that reflect {director_style}'s signature techniques
                    - Consider how {director_style} would frame shots and direct actors

                    Now write a compelling scene that meets ALL specifications exactly, focusing on meeting the word count target in the upper range.
                    """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def write_scene(self, genre: str, setting: str, idea: str, 
                   director_style: str, length: str, examples: List[str]) -> str:
        length = length.lower()
        requirements = self.length_requirements.get(length, self.length_requirements["medium"])
        examples_text = "\n\n".join(examples) if examples else "No examples provided."
        
        return self.chain.run(
            genre=genre,
            setting=setting,
            idea=idea,
            director_style=director_style,
            length=length,
            examples=examples_text,
            expected_word_count=f"{requirements['min_words']}-{requirements['max_words']}",
            expected_paragraphs=f"{requirements['min_paragraphs']}-{requirements['max_paragraphs']}",
            expected_exchanges=f"{requirements['min_exchanges']}-{requirements['max_exchanges']}",
            length_description=requirements['description']
        )

class EditorAgent:
    def __init__(self, temperature: float = 0.4):  
        self.llm = ChatOpenAI(temperature=temperature)
        self.expansion_llm = ChatOpenAI(temperature=0.7) #can adjust temp if needed
        self.length_requirements = LENGTH_REQUIREMENTS

    
        self.prompt = PromptTemplate(
            input_variables=[
                "scene", "genre", "director_style", "length", "min_words", "max_words",
                "min_paragraphs", "max_paragraphs", "min_exchanges", "max_exchanges",
                "length_description", "length_display", "current_words", 
                "current_paragraphs", "current_exchanges"
            ],
            template="""
                    You are a masterful script editor with a talent for enhancing scenes to their full potential. Your task is to expand and enrich the following scene to meet specific length and format requirements.

                    ORIGINAL SCENE:
                    {scene}

                    ------------------
                    TARGET LENGTH: {length_display}
                    - Word count: {min_words}-{max_words} (currently: {current_words}) - AIM FOR THE UPPER RANGE!
                    - Paragraphs: {min_paragraphs}-{max_paragraphs} (currently: {current_paragraphs})
                    - Dialogue exchanges: {min_exchanges}-{max_exchanges} (currently: {current_exchanges})
                    - Description: {length_description}
                    ------------------

                    EXPANSION REQUIREMENTS:
                    1. ADD SUBSTANTIAL CONTENT to reach the target word count in the upper range
                    2. Maintain the scene's core premise while significantly enhancing:
                       - Scene descriptions with rich visual and sensory details
                       - Character development through actions and reactions
                       - Dialogue depth and complexity
                       - Emotional beats and subtext
                       - Environmental details that create atmosphere
                    3. Reflect {director_style}'s visual style and cinematic approach
                    4. Emphasize genre-specific elements of {genre}
                    5. Maintain proper screenplay format throughout

                    IMPORTANT INSTRUCTION:
                    For scenes marked {length_display}, be GENEROUS with your expansion. Add memorable moments, character quirks, and vivid descriptions that make the scene distinctive.

                    Return ONLY the fully expanded scene in perfect screenplay format.
                    """
            )

        self.expansion_prompt = PromptTemplate(
            input_variables=["scene", "genre", "director_style", "min_words", "max_words", "current_words", "target_words"],
            template="""
                EMERGENCY EXPANSION REQUIRED! The scene is significantly under the word count target.

                ORIGINAL SCENE:
                {scene}

                CURRENT STATUS:
                - Current word count: {current_words}
                - TARGET word count: {target_words} (at least {min_words}, up to {max_words})
                - YOU MUST ADD AT LEAST {target_words} MORE WORDS to this scene!

                MAJOR EXPANSION REQUIREMENTS:
                1. Double or triple the scene's length while maintaining quality and coherence
                2. Add substantial new content in these areas:
                   - 3-5 new paragraphs of vivid scene description
                   - At least 4-6 new dialogue exchanges between characters
                   - Introduce a new emotional beat or complication
                   - Expand existing dialogue to show character depth
                   - Add a new dramatic moment or revelation
                3. Incorporate {genre} genre elements and {director_style}'s signature visual style
                4. Maintain proper screenplay format throughout

                EXPANSION STRATEGIES:
                - Introduce a new complication or obstacle for characters
                - Add a revealing character moment or backstory hint
                - Develop a secondary conflict or subplot element
                - Explore character relationships more deeply
                - Add sensory details that enhance the atmosphere
                - Include more specific action descriptions

                Return ONLY the fully expanded scene in perfect screenplay format, ensuring it meets the target word count.
                """
        )

        self.validation_prompt = PromptTemplate(
            input_variables=["scene", "genre", "director_style", "min_words", "max_words", "current_words", "target_words", "deficit"],
            template="""
                    FINAL REVIEW AND EXPANSION NEEDED

                    The scene is still {deficit} words short of the minimum target. Your task is to make one final expansion while maintaining quality.

                    CURRENT SCENE:
                    {scene}

                    OBJECTIVE:
                    - Current word count: {current_words}
                    - Minimum required: {min_words}
                    - Target word count: {target_words}
                    - Maximum allowed: {max_words}
                    - ADD AT LEAST {deficit} MORE WORDS to reach the minimum!

                    FINAL EXPANSION FOCUS:
                    1. Add 1-2 paragraphs of rich visual description
                    2. Enhance dialogue with more character-specific language
                    3. Add sensory details that reflect {director_style}'s style
                    4. Include elements typical of {genre} that enrich the scene

                    Maintain perfect screenplay format and ensure the additions feel seamless and integral to the scene.

                    Return ONLY the final expanded scene.
                    """
            )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.expansion_chain = LLMChain(llm=self.expansion_llm, prompt=self.expansion_prompt)
        self.validation_chain = LLMChain(llm=self.llm, prompt=self.validation_prompt)

    def count_metrics(self, scene: str):
        words = len(scene.split())
        paragraphs = len([p for p in scene.split("\n\n") if p.strip()])
        exchanges = len(re.findall(r'^[A-Z ]+:', scene, re.MULTILINE))
        return words, paragraphs, exchanges

    def edit_scene(self, scene: str, genre: str, director_style: str, length: str) -> str:
        length = length.lower()
        requirements = self.length_requirements.get(length, self.length_requirements["medium"])

        word_count, paragraph_count, exchange_count = self.count_metrics(scene)

        # Calculate target word count (aim for the middle-upper part of the range)
        target_words = int((requirements["min_words"] + requirements["max_words"]) * 0.75)

        # First pass - basic editing
        edited_scene = self.chain.run(
            scene=scene,
            genre=genre,
            director_style=director_style,
            length=length,
            min_words=requirements["min_words"],
            max_words=requirements["max_words"],
            min_paragraphs=requirements["min_paragraphs"],
            max_paragraphs=requirements["max_paragraphs"],
            min_exchanges=requirements["min_exchanges"],
            max_exchanges=requirements["max_exchanges"],
            length_description=requirements["description"],
            length_display=requirements["display"],
            current_words=word_count,
            current_paragraphs=paragraph_count,
            current_exchanges=exchange_count
        )
        
        # Check if it meets requirements
        new_word_count, new_paragraph_count, new_exchange_count = self.count_metrics(edited_scene)
        
        # If still too short, do aggressive expansion
        if new_word_count < requirements["min_words"]:
            expanded_scene = self.expansion_chain.run(
                scene=edited_scene,
                genre=genre,
                director_style=director_style,
                min_words=requirements["min_words"],
                max_words=requirements["max_words"],
                current_words=new_word_count,
                target_words=target_words
            )
            
            # Final validation check
            final_word_count, _, _ = self.count_metrics(expanded_scene)
            
            # If still below minimum, do one more focused expansion
            if final_word_count < requirements["min_words"]:
                deficit = requirements["min_words"] - final_word_count
                validated_scene = self.validation_chain.run(
                    scene=expanded_scene,
                    genre=genre,
                    director_style=director_style,
                    min_words=requirements["min_words"],
                    max_words=requirements["max_words"],
                    current_words=final_word_count,
                    target_words=target_words,
                    deficit=deficit
                )
                return validated_scene
            
            return expanded_scene
        
        return edited_scene

class ScriptGenerationCrew:
    def __init__(self):
        self.rag_agent = RAGAgent()
        self.writer_agent = WriterAgent()
        self.editor_agent = EditorAgent()
        
    def initialize_with_data(self, data_dir: str):
        """Initialize the RAG agent with annotated scripts"""
        self.rag_agent.initialize_vector_store(data_dir)
    
    # IMPROVED: Enhanced generate_scene with length validation
    def generate_scene(self, 
                      genre: str, 
                      setting: str, 
                      idea: str,
                      director_style: str,
                      length: str,
                      debug: bool = False) -> Dict[str, str]:
        """Generate a complete scene using all agents"""
        # Validate and standardize length parameter
        length = length.lower().strip()
        if length not in LENGTH_REQUIREMENTS:
            length = "medium"
        
        requirements = LENGTH_REQUIREMENTS[length]
        
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
        
        # 3. Edit the scene
        final_scene = self.editor_agent.edit_scene(
            scene=initial_scene,
            genre=genre,
            director_style=director_style,
            length=length
        )
        
        # Calculate metrics
        def count_scene_elements(scene: str) -> dict:
            paragraphs = len([p for p in scene.split('\n\n') if p.strip() and not p.strip().startswith(('INT.', 'EXT.'))])
            dialogue_exchanges = len([d for d in scene.split('\n\n') if any(c.isupper() for c in d[:20])])
            return {
                'word_count': len(scene.split()),
                'paragraphs': paragraphs,
                'dialogue_exchanges': dialogue_exchanges
            }
        
        initial_metrics = count_scene_elements(initial_scene)
        final_metrics = count_scene_elements(final_scene)
        
        # NEW: Final validation check to ensure minimum length requirements are met
        final_word_count = final_metrics['word_count']
        
        # If still below target, try one more expansion with a stronger instruction
        if final_word_count < requirements["min_words"] and length != "short":
            print(f"Warning: Scene still below minimum length ({final_word_count}/{requirements['min_words']}). Attempting emergency expansion...")
            
            # Create a special expansion prompt for severe cases
            emergency_expansion_llm = ChatOpenAI(temperature=0.8)  # Higher creativity
            emergency_prompt = PromptTemplate(
                input_variables=["scene", "min_words", "current_words"],
                template="""
                CRITICAL LENGTH ISSUE: The screenplay scene is severely under the required length.
                
                Current scene ({current_words} words):
                {scene}
                
                YOUR TASK:
                Transform this scene into a MUCH LONGER VERSION (at least {min_words} words) by:
                1. Adding substantially more description to every paragraph
                2. Creating new dialogue exchanges
                3. Adding atmospheric elements and sensory details
                4. Expanding character actions and reactions
                5. Including more visual direction
                
                IMPORTANT: DOUBLE OR TRIPLE the length while maintaining screenplay format!
                
                Return the COMPLETE expanded scene only.
                """
            )
            
            emergency_chain = LLMChain(llm=emergency_expansion_llm, prompt=emergency_prompt)
            final_scene = emergency_chain.run(
                scene=final_scene,
                min_words=requirements["min_words"],
                current_words=final_word_count
            )
            
            # Recalculate final metrics after emergency expansion
            final_metrics = count_scene_elements(final_scene)
        
        # Return comprehensive results
        result = {
            "initial_scene": initial_scene,
            "final_scene": final_scene,
            "examples_used": relevant_examples if debug else [],
            "metrics": {
                "length_setting": length,
                "target_min_words": requirements["min_words"],
                "target_max_words": requirements["max_words"],
                "initial_word_count": initial_metrics['word_count'],
                "final_word_count": final_metrics['word_count'],
                "word_count_change": final_metrics['word_count'] - initial_metrics['word_count'],
                "initial_paragraphs": initial_metrics['paragraphs'],
                "final_paragraphs": final_metrics['paragraphs'],
                "initial_dialogue_exchanges": initial_metrics['dialogue_exchanges'],
                "final_dialogue_exchanges": final_metrics['dialogue_exchanges'],
                "meets_requirements": final_metrics['word_count'] >= requirements["min_words"]
            }
        }
        
        if not debug:
            result.pop("examples_used", None)
            
        return result

if __name__ == "__main__":
    crew = ScriptGenerationCrew()
    
    crew.initialize_with_data(
        data_dir="data"
    )
    
    # Generate a scene
    result = crew.generate_scene(
        genre="Science Fiction",
        setting="Space Station",
        idea="A crew member discovers an alien artifact that seems to be alive",
        director_style="Steven Spielberg",
        length="short",
        debug=False
    )
    
    print("Final Scene:")
    print(result["final_scene"])

    #if u want to see metrics
    # print("\nMetrics:")
    # for k, v in result["metrics"].items():
    #     print(f"{k}: {v}")