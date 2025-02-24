import streamlit as st
from transformers import pipeline


# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    return pipeline("token-classification", model="dslim/bert-base-NER")


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        dict: Transcription result with text and optional timestamps.
    """
    # Load the Whisper model
    speech_to_text = load_whisper_model()

    # Read the file directly from the uploaded file object
    audio_bytes = uploaded_file.read()

    # Transcribe the audio using the model
    result = speech_to_text(audio_bytes, return_timestamps=True)
    return result


# ------------------------------
# Entity Extraction
# ------------------------------

def extract_entities(text):
    """
    Extract entities from text using NER with aggregation strategy.
    Args:
        text (str): Input text.
    Returns:
        dict: Grouped entities by type (PER, ORG, LOC, etc.).
    """
    # Load the NER model
    ner_pipeline = load_ner_model()

    # Run NER pipeline with simple aggregation strategy
    results = ner_pipeline(text, aggregation_strategy="simple")

    # Group entities by type
    grouped_entities = {}
    for entity in results:
        entity_type = entity["entity_group"]
        entity_text = entity["word"]

        # Add entity to the corresponding group
        if entity_type not in grouped_entities:
            grouped_entities[entity_type] = []
        if entity_text not in grouped_entities[entity_type]:
            grouped_entities[entity_type].append(entity_text)

    return grouped_entities


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "MHD LAITH ALKURDI"
    STUDENT_ID = "150230909"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    st.markdown("""
        Upload a business meeting audio file to:
        1. Transcribe the meeting audio into text.
        2. Extract key entities such as Persons, Organizations, Dates, and Locations.
        """)

    # File uploader section
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"], help="Limit 200MB per file")

    # Process uploaded file
    if uploaded_file is not None:
        # Show loading state
        with st.spinner("Transcribing the audio file... This may take a minute."):
            transcribed_text = transcribe_audio(uploaded_file)["text"]

        # Transcription section
        if transcribed_text:
            st.success("Transcription Complete!")
            st.write("### Transcription:")
            st.write(transcribed_text)

            # Extract entities
            with st.spinner("Extracting entities..."):
                entities = extract_entities(transcribed_text)

            # Display extracted entities
            st.write("### Extracted Entities:")

            # ORGs
            st.markdown("#### Organizations (ORGs):")
            for org in entities.get("ORG", []):
                st.write(f"- {org}")

            # LOCs
            st.markdown("#### Locations (LOCs):")
            for loc in entities.get("LOC", []):
                st.write(f"- {loc}")

            # PERs
            st.markdown("#### Persons (PERs):")
            for person in entities.get("PER", []):
                st.write(f"- {person}")



if __name__ == "__main__":
    main()
