import os

def get_answer_llm():

    from langchain_google_genai import ChatGoogleGenerativeAI
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key = google_api_key,
        temperature=0,
    )
