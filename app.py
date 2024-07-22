import transformers
import torch
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

def load_model_tokenizer(repository):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        repository,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(repository)
    return model, tokenizer

def get_response(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_message = "You are a world class fitness instructor and gym trainer. You will give proper exercise and diet plans if asked. Always answer the user in detail and in bullet points."
    prompt = f"system: {system_message} user: {text} assistant:"
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    return output.split("\n")

def process_video(video_path):
    # Placeholder function to process the video
    # Here you can implement the logic for fitness tracking, e.g., pose detection
    return "Video processing is not yet implemented."

st.set_page_config(page_title='Fitness Instructor', page_icon="🏃‍♂️")
st.title("Fitness Instructor")

## Creating the chat_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your Fitness Instructor. I will do my best to help you to the best of my abilities.")
    ]

user_query = st.chat_input('Enter your query here...')

if user_query is not None and user_query != "":
    model, tokenizer = load_model_tokenizer("AdityaLavaniya/TinyLlama-Fitness-Instructor")
    response = get_response(user_query, model, tokenizer)

    # Updating the chat_history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    ## Displaying the chat_history in Application
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

st.header("Fitness Tracker")

option = st.selectbox("Choose an option", ["Upload Video", "Live Video"])

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload your video here", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        # Display the uploaded video
        st.video(temp_video_path)

        # Process video for fitness tracking
        results = process_video(temp_video_path)
        st.write(results)

        # Clean up the temporary file
        os.remove(temp_video_path)

elif option == "Live Video":
    st.write("Starting live video feed...")
    run = st.checkbox('Run')

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
