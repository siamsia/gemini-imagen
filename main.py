from fastapi import FastAPI, Response
from pydantic import BaseModel
from google.generativeai import GenerativeModel
from google.generativeai.types import GenerationConfig, ResponseModality
from PIL import Image
from io import BytesIO
import os

app = FastAPI()

# ให้เซี้ยตั้งค่า API Key เป็น Environment Variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# สร้าง GenerativeModel และกำหนดค่า responseModality เพื่อให้สร้างภาพได้
model = GenerativeModel(
    model_name="gemini-2.0-flash-preview-image-generation",
    api_key=API_KEY,
    generation_config=GenerationConfig(
        response_modalities=[ResponseModality.TEXT, ResponseModality.IMAGE]
    )
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-image/")
async def generate_image(request: PromptRequest):
    """
    Endpoint สำหรับสร้างภาพจากข้อความ Prompt
    """
    try:
        response = model.generate_content(request.prompt)

        # ตรวจสอบว่ามีภาพในผลลัพธ์หรือไม่
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.is_image_part():
                    # แปลงภาพเป็นไบต์
                    image_bytes = BytesIO()
                    part.image.save(image_bytes, format='PNG')
                    image_bytes.seek(0)
                    
                    return Response(content=image_bytes.getvalue(), media_type="image/png")

        return {"error": "ไม่พบภาพในผลลัพธ์"}

    except Exception as e:
        return {"error": f"เกิดข้อผิดพลาด: {e}"}

