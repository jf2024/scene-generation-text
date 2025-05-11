# Scene-Generation-Text
By: Maxwell, Arjun, and Jose


This deep learning project enables users to generate professional-quality screenplay scenes based on specified genres, settings, core ideas, and directorial styles. Using a sophisticated multi-agent system, it creates original screenplay content that captures the essence of renowned directors and adheres to industry-standard formatting.

<img width="1506" alt="Screenshot 2025-05-11 at 4 04 27 PM" src="https://github.com/user-attachments/assets/8acc3cce-dbd0-4e84-b0a0-64af1a4e35a0" />


- Generate original screenplay scenes that emulate the style of specific directors
- Create properly formatted screenplay content with scene headings, descriptions, and dialogue
- Provide customizable options for genre, setting, length, and creative direction
- Implement smart length control to meet specific word count requirements
- Deliver consistent quality through advanced editing and refinement

## Technical Architecture

The system employs three specialized AI agents working together:

### RAG Agent

The Retrieval-Augmented Generation (RAG) agent leverages a curated database of annotated screenplay examples:

- Indexes script examples with metadata (genre, director, plot, cast)
- Uses semantic search to find relevant references matching user requirements
- Retrieves contextually appropriate examples to guide generation

### Writer Agent

Creates initial screenplay drafts based on user inputs and reference materials:

- Follows proper screenplay formatting conventions
- Generates scene headings, descriptions, and dialogue
- Implements directorial style elements
- Adheres to length requirements

### Editor Agent

Refines and enhances initial drafts through multiple improvement passes:

- Ensures consistent formatting and style
- Expands content to meet minimum length requirements
- Enhances dialogue and scene descriptions
- Performs emergency expansion for severely short drafts

## Features

- **Style Emulation:** Capture the distinctive voice and visual style of different directors
- **Customizable Length:** Generate scenes in four different size categories:
  - **Short:** Brief moment (200-250 words)
  - **Medium:** Complete scene (400-550 words)
  - **Long:** Detailed scene (800-1200 words)
- **Multi-Stage Generation:** Initial drafting followed by specialized editing
- **Quality Control:** Rigorous validation checks ensure high-quality output

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scene-generation-text.git
   cd scene-generation-text
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. Enter your script idea in the text field
2. Specify the setting for your scene
3. Select a genre from the dropdown menu
4. Choose a director style (e.g., Quentin Tarantino)
5. Select your desired scene length
6. Click "Generate" to create your screenplay scene

## 🗂️ Project Structure

- `app.py` - Streamlit application front-end
- `final_agent.py` - Main agent implementation with dialogue enhancements
- `requirements.txt` - Required Python packages
- `agents_v1.ipynb` - Initial notebook implementation, testing phase
- `agents_v2.py` - original agent implementation, no dialogue or prompt enhancements
- `finalScriptGeneration.ipynb` - Final implementation with improved package integration
- `explore2.ipynb` - Data exploration notebook along with data cleaning and preprocessing

## 🔜 Future Development

- Character memory across multiple scenes
- Visual storyboarding capabilities
- Specialized genre-specific models
- Expanded directorial style library
