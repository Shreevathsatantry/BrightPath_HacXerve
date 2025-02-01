from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from flask_cors import CORS
import google.generativeai as genai
import re
from PIL import Image
import base64
from io import BytesIO
import time
import requests
import io
from huggingface_hub import InferenceClient
from audio_extract import extract_audio
import os
from deep_translator import GoogleTranslator


app = Flask(__name__)
CORS(app)

client = InferenceClient("stabilityai/stable-diffusion-3.5-large-turbo", token="")

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

@app.route("/StoryTeller", methods=["POST"])
def storyTeller():
    input_data = request.get_json()
    input_text = input_data.get("text", "")
    
    response = model.generate_content(f"Tell a children's story based on the given context in 4 paragraphs: {input_text}")

    ImageGen(response.text)
    
    return jsonify({"response": response.text})


def ImageGen(text):
    ParaList = text.split("\n\n")
    for i in range(len(ParaList) - 1):
        PromptImage = model.generate_content(f"Generate a scenario based prompt no more than 20 words to generate an image based on the following context: {ParaList[i]}")
        print(PromptImage.text)
        image = client.text_to_image(PromptImage.text)
        image.save(f"public/Images/Image{i + 1}.png")
    
    client_1 = InferenceClient("stabilityai/stable-diffusion-3.5-large-turbo", token="")
    PromptImage = model.generate_content(f"Generate a scenario based very short prompt to generate an image based on the following context: {ParaList[3]}")
    image = client_1.text_to_image(PromptImage.text)
    image.save(f"public/Image4.png")
    # print(ParaList[0])

@app.route("/QuizBot", methods=["POST"])
def quizBot():
    input_data = request.get_json()
    input_text = input_data.get("text", "")

    text = """
    Generate 10 multiple-choice questions based on the provided story. For each question, include:
    1. The question text.
    2. Four options (labeled a, b, c, d).
    3. skip 2 lines before starting the next question.

    Use this exact format for the response:
    Question 1: [Your question here]
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]
    Correct Answer: [Letter of the correct answer]

    Repeat for all 10 questions.
    """

    # Request the model to generate questions in the defined format
    response = model.generate_content(f"in the following format: {text} \n Frame 10 questions and give 4 options with one correct answer on the following story: {input_text}")
    print(response.text)

    # Now, parse the model's response into a structured format
    quiz_data = parse_quiz_response(response.text)
    print(quiz_data)
    
    return jsonify({"response": quiz_data})

# Helper function to parse the response into structured questions, options, and correct answers
def parse_quiz_response(response):
    # Split the response by questions using regular expressions to capture question text
    questions = re.split(r"(?=Question \d+:)", response.strip())

    parsed_questions = []

    for question in questions:
        lines = question.strip().split("\n")
        
        if len(lines) >= 6:
            # Extract the question, options, and the correct answer
            question_text = lines[0].replace("Question \d+: ", "").strip()
            
            # Retain the labels a), b), c), d) for options
            options = [line.strip() for line in lines[1:5]]
            
            # Extract the correct answer, removing the label like 'Correct Answer: b'
            correct_answer = lines[5].replace("Correct Answer: ", "").strip()

            parsed_questions.append({
                "question": question_text,
                "options": options,
                "correctAnswer": correct_answer
            })

    return parsed_questions

@app.route("/LearnBot", methods=["POST"])
def learnBot():
    input_data = request.get_json()
    input_text = input_data.get("text", "")
    image_base64 = input_data.get("image", "")

    time.sleep(3)

    downloads_path = get_downloads_folder()
    file_name = "audio.wav"
    audio_file_path = os.path.join(downloads_path, file_name)

    if os.path.exists(audio_file_path):

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model_analyze = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )

        files_audio = [
            upload_to_gemini(audio_file_path, mime_type="audio/mp3"),
        ]

        file_uri_audio = files_audio[0]

        response_audio = model_analyze.generate_content(
            ["What is said here?", file_uri_audio],
            request_options={"timeout": 600}
        )

        print(response_audio.text)

        input_text = response_audio.text

        os.remove(audio_file_path)

    image = None
    if image_base64:
        try:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]

            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            print("Error decoding or verifying image:", e)
            return jsonify({"error": "Invalid image data"}), 400

    try:
        if image and input_text:
            prompt = f"Explain this image and the following text in a simple way for children: {input_text}"
            response = model.generate_content([prompt, image])
        elif image:
            prompt = "Explain this image in a simple way for children."
            response = model.generate_content([prompt, image])
        elif input_text:
            prompt = f"Analyze the input text and detect its language. Text Respond in the same language by explaining the text in a simple way suitable for children. Ensure the explanation is clear, concise, and easy to understand also do not give long responses.: {input_text}"
            response = model.generate_content(prompt)
        else:
            return jsonify({"error": "No valid input provided"}), 400

        return jsonify({"response": response.text})
    except Exception as e:
        print("Error generating response:", e)
        return jsonify({"error": "Failed to generate response"}), 500

@app.route("/AiSuggestionBot", methods=["GET"])
def aiSuggestionBot():

    text = """
    Language Development: Language Development suggestion,
    Physical Development: Physical Development suggestion,
    Cognitive Skills: Cognitive Skills suggestion,
    Communication Skills: Communication Skills suggestion,
    """
    
    response = model.generate_content(f"Give a 4 brief suggestions for the parents on how to improve their childs Language Development, Physical Development, Cognitive Skills, communication skills in the form of: {text}")
    return jsonify({"response": response.text})

def get_downloads_folder():
    """Get the path to the Downloads folder."""
    home = os.path.expanduser("~")
    downloads_folder = os.path.join(home, "Downloads")
    return downloads_folder

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def translate_me_baby(text, lan):
    translated = GoogleTranslator(source='auto', target=f"{lan}").translate(text)
    return translated


@app.route("/VideoAnalyzer", methods=["GET"])
def videoAnalyzer():
    downloads_path = get_downloads_folder()
    file_name = "Video.mp4"
    video_file_path = os.path.join(downloads_path, file_name)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-2.0-flash-exp",
      generation_config=generation_config,
    )

    print(f"Checking for file {file_name} in Downloads folder...")

    # Loop until the file is available
    while not os.path.exists(video_file_path):
        print(f"Waiting for {file_name} to be available...")
        time.sleep(5)

    print(f"File {file_name} found. Uploading...")

    extract_audio(input_path=video_file_path, output_path="./audio.mp3")

    files_audio = [
        upload_to_gemini("audio.mp3", mime_type="audio/mp3"),
    ]
    file_uri_audio = files_audio[0]

    response_audio = model.generate_content(
        ["Only tell what is said here?", file_uri_audio],
        request_options={"timeout": 600}
    )
    user_audio = response_audio.text

    model = genai.GenerativeModel("gemini-1.5-flash")

    LangFormat = """
    Language: "[language]"
    """
   
    response1 = model.generate_content(f"Which language is this: {response_audio.text}. Answer in this format: {LangFormat}")
    language = response1.text.strip()

    print("Answer: ", language)

    video_file = genai.upload_file(path=video_file_path)
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    
    print(f'Video processing complete: ' + video_file.uri)
    Format = """
    response : [response]
    """     

    prompt = f"""
    give appropriate response to this question in this format {Format} The question: "{response_audio.text}". Respond in english
    """

    # Make the LLM request.
    print("Making LLM inference request...")

    response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
    print(response.text)
    
    if "Kannada" in language:
        target_language = "kn"
        print("Translating to Kannada")
    elif "Hindi" in language:
        target_language = "hi"
        print("Translating to Hindi")
    elif "English" in language:
        target_language = "en"
        print("Translating to English")
    elif "Marathi" in language:
        target_language = "mr"
        print("Translating to Marathi")
    else:
        target_language = "en"  # Default to English
        print(f"Language '{language}' not explicitly supported. Defaulting to English.")
    print(target_language)
    response = translate_me_baby(response.text , target_language )
    
    print(response)
    
    os.remove(video_file_path)
    os.remove("./audio.mp3")
    print(LangFormat[14:-2])
    
    return jsonify({"response": response[response.find(':')+1:] , "user_audio":user_audio})

if __name__ == "__main__":
    app.run(debug=True)