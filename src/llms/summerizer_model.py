from langchain_google_genai import ChatGoogleGenerativeAI
from config import envConfig


summarizer = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=envConfig.GOOGLE_API_KEY
)