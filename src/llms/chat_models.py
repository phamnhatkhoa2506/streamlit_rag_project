from langchain_google_genai import ChatGoogleGenerativeAI
from config import envConfig


main_chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    api_key=envConfig.GOOGLE_API_KEY,
)