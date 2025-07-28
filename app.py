import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
from googletrans import Translator, LANGUAGES
import PyPDF2
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom CSS for enhanced styling
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.stApp {
    background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
}
.stButton>button {
    color: black;
    background-color: #4CAF50;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    transition-duration: 0.4s;
    cursor: pointer;
    border-radius: 12px;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
}
</style>
""", unsafe_allow_html=True)

# Summary Styles
SUMMARY_STYLES = {
    "Normal": """You are an advanced content summarizer. Provide a balanced, straightforward summary in clear points within 250 words. Include:
1. Main topic of the content
2. Key points discussed
3. Significant insights
4. Overall purpose of the content""",
    
    "Concise": """Provide an extremely brief and to-the-point summary. Capture the core message in the most compact form possible, within 150 words. Focus on:
1. Primary topic
2. Most critical takeaways
3. Absolute key points""",
    
    "Explained": """Create a detailed, in-depth summary that breaks down the content with comprehensive explanations. Aim for 350-400 words. Include:
1. Detailed context
2. Comprehensive explanation of key concepts
3. In-depth insights and nuanced interpretations
4. Broader implications of the discussed topics""",
    
    "Formal": """Generate an academically structured and professionally articulated summary. Maintain a scholarly tone and precise language within 250 words. Address:
1. Scholarly interpretation of the subject
2. Analytical breakdown of key arguments
3. Theoretical or professional insights
4. Precise contextual analysis"""
}

# Translator setup
translator = Translator()

# Language configurations
CUSTOM_LANGUAGES = {
    'kn': 'Kannada', 'hi': 'Hindi', 
    'te': 'Telugu', 'ml': 'Malayalam', 
    'ta': 'Tamil'
}
LANGUAGES.update(CUSTOM_LANGUAGES)

def translate_text(text, target_lang):
    """
    Translate text to target language
    """
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def extract_transcript_details(youtube_video_url):
    """
    Multiple methods to extract video text content
    """
    try:
        # Extract video ID
        video_id = youtube_video_url.split("=")[-1]
        
        # Method 1: Try getting transcript
        try:
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([entry['text'] for entry in transcript_text])
            if transcript.strip():
                return transcript
        except Exception:
            pass
        
        # Method 2: Try video description
        try:
            yt = YouTube(youtube_video_url)
            description = yt.description or ""
            if description.strip():
                return description
        except Exception:
            pass
        
        # Method 3: Web scraping for video title and description
        try:
            response = requests.get(youtube_video_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('meta', property='og:title')
            description = soup.find('meta', property='og:description')
            
            fallback_text = ""
            if title:
                fallback_text += title.get('content', '') + " "
            if description:
                fallback_text += description.get('content', '')
            
            return fallback_text.strip()
        except Exception:
            pass
        
        return "No transcript or description available for this video."
    
    except Exception:
        return "Unable to process the video link."

def extract_text_from_pdf(pdf_file):
    """
    Extract text content from PDF file
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return "Could not extract text from PDF"

def extract_text_from_image(image_file):
    """
    Extract text from image using pytesseract OCR with robust error handling
    """
    try:
        # Convert image to numpy array
        if isinstance(image_file, Image.Image):
            # If it's already a PIL Image, convert directly
            img = np.array(image_file)
        else:
            # If it's a file upload, open with PIL first
            img = Image.open(image_file)
            img = np.array(img)
        
        # Ensure the image is in the right color format for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optional: Apply image preprocessing to improve OCR accuracy
        # Threshold the image to make text more clear
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(thresh)
        
        return text.strip() if text else "No text could be extracted from the image"
    
    except Exception as e:
        # Log the error for debugging, but don't show it to the user
        st.warning("Could not extract text from image. Please try a different image.")
        return "No text could be extracted from the image"

def extract_text_from_video(video_file):
    """
    Extract frames and text from video
    """
    try:
        # Open the video file
        video = cv2.VideoCapture(video_file.name)
        
        # Extract text from first few frames
        text_snippets = []
        frame_count = 0
        while frame_count < 10:  # Extract text from first 10 frames
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract text
            frame_text = pytesseract.image_to_string(gray)
            if frame_text.strip():
                text_snippets.append(frame_text)
            
            frame_count += 1
        
        video.release()
        return " ".join(text_snippets) if text_snippets else "No text extracted from video"
    except Exception as e:
        st.error(f"Video text extraction error: {e}")
        return "Could not extract text from video"

def generate_gemini_content(text, prompt):
    """
    Generate summary using Gemini Pro
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + text)
        return response.text
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")
        return "Could not generate summary due to an error"

def main():
    st.markdown("""
    <h1 style="text-align: center; color: white;">🎥 AI Powered Video Summarizer</h1>
    """, unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Select Input Method", 
        ["YouTube Link", "Upload PDF", "Upload Image"],
        horizontal=True
    )

    # Initialize input variable
    input_content = None

    # Input selection based on method
    if input_method == "YouTube Link":
        input_content = st.text_input("🔗 Enter YouTube Video Link", placeholder="https://www.youtube.com/watch?v=...")
    elif input_method == "Upload PDF":
        input_content = st.file_uploader("📄 Upload PDF", type=['pdf'])
    elif input_method == "Upload Image":
        input_content = st.file_uploader("🖼️ Upload Image", type=['png', 'jpg', 'jpeg'])
    
    # Language and summary style columns
    col1, col2 = st.columns(2)
    
    with col1:
        summary_style = st.selectbox(
            "📝 Summary Style", 
            list(SUMMARY_STYLES.keys()),
            index=0
        )
    
    with col2:
        languages_list = [(code, name) for code, name in LANGUAGES.items() if code != 'auto']
        if 'en' not in [code for code, _ in languages_list]:
            languages_list.append(('en', 'English'))
        
        sorted_languages = sorted(languages_list, key=lambda x: x[1])
        selected_language = st.selectbox(
            "🌐 Select Translation Language", 
            [f"{name} ({code})" for code, name in sorted_languages],
            index=next((i for i, (code, _) in enumerate(sorted_languages) if code == 'en'), 0)
        )

    # Generate Summary button
    generate_summary = st.button("Generate Summary 🚀")

    # Process content and generate summary
    if generate_summary and input_content:
        # Extract language code
        selected_lang_code = selected_language.split('(')[-1].strip(')')

        # Progress indicator
        with st.spinner('Generating Summary...'):
            # Extract text based on input method
            if input_method == "YouTube Link":
                if "youtube.com/watch?v=" not in input_content:
                    st.error("Please enter a valid YouTube video link!")
                    return
                video_text = extract_transcript_details(input_content)
            elif input_method == "Upload PDF":
                video_text = extract_text_from_pdf(input_content)
            elif input_method == "Upload Image":
                video_text = extract_text_from_image(input_content)
           

            # Generate summary with selected style
            summary = generate_gemini_content(video_text, SUMMARY_STYLES[summary_style])
            
            # Translate summary if not English
            if selected_lang_code != 'en':
                translated_summary = translate_text(summary, selected_lang_code)
            else:
                translated_summary = summary
            
            # Display summary
            st.markdown(f"## 📌 {summary_style} Summary ({selected_language})")
            st.markdown(f"""
            <div style="
                background-color:#f1f8ff; 
                padding:20px; 
                border-radius:10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                color: #333333; 
                font-family: Arial, sans-serif; 
                line-height: 1.6;">
            {translated_summary}
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Google Gemini AI, Google Translate* 🤖✨")

# Run the main function
if __name__ == "__main__":
    main()