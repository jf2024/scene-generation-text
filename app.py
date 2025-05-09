import streamlit as st
from agents_v2 import ScriptGenerationCrew

st.title("Script Generator")

prompt = st.text_input("Enter your script idea")

setting = st.text_input("Enter the setting for the script")

genre = st.selectbox(
    "Choose a genre for the script",
    ["Action", "Comedy", "Romance", "Horror", "Thriller"],
    placeholder="Action"
)

director = st.selectbox(
    "Choose a directing style for the script",
    ["Quentin Tarantino", "Steven Spielberg", "Stanley Kubrick", "Alfred Hitchcock", "Tim Burton"],
    placeholder="Quentin Tarantino"
)

scene_length = st.selectbox(
    "Select the length of the scene you want to generate",
    ["Short", "Medium", "Long"],
    placeholder="Short"
)


if st.button("Generate"):
    st.write("Prompt:", prompt)
    st.write("Genre:", genre)
    st.write("Director Style:", director)
    st.write("Scene Length:", scene_length)

    crew = ScriptGenerationCrew()

    crew.initialize_with_data(
        data_dir="data"
    )

    result = crew.generate_scene(
        genre=genre,
        setting=setting,
        idea=prompt,
        director_style=director,
        length=scene_length
    )

    st.write(result)
