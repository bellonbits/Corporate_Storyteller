from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import uuid
import datetime
import requests
from typing import List, Optional, Dict, Any
from fpdf import FPDF
import io
import textwrap

# Corporate Branding Colors
CORPORATE_COLORS = {
    "primary": "#0F4C81",
    "secondary": "#4A90E2",
    "accent": "#F2994A",
    "success": "#6FCF97",
    "warning": "#F2C94C",
    "danger": "#EB5757",
    "neutral": "#606060",
    "background": "#F8F9FA"
}

# Groq API Settings
GROQ_API_KEY = "your_groq_api_key_here"  # Update this!
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Instantiate App
app = FastAPI(title="Corporate Data Storyteller")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Models
class StoryGenerationRequest(BaseModel):
    corporate_style: str
    business_context: str
    audience: str
    title: Optional[str] = None

class StoryResponse(BaseModel):
    report_id: str
    title: str
    preview: str
    pdf_url: str

# Helper Functions
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# PDF Report Class
class CorporateReportPDF(FPDF):
    def __init__(self, title="Business Data Analysis", company_name="Your Company"):
        super().__init__()
        self.title = title
        self.company_name = company_name
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_fill_color(15, 76, 129)
        self.rect(0, 0, 210, 30, style="F")
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 24)
        self.cell(0, 15, self.title, ln=True, align="C")
        self.set_font("Arial", "I", 12)
        self.cell(0, 10, self.company_name, ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def section_title(self, title):
        self.set_font("Arial", "B", 16)
        self.set_fill_color(74, 144, 226)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.set_font("Arial", "", 11)
        self.ln(5)

    def section_body(self, body):
        self.set_font("Arial", "", 11)
        for line in textwrap.wrap(body, 80):
            self.cell(0, 6, line, ln=True)
        self.ln(5)

    def add_bullet_points(self, points):
        self.set_font("Arial", "", 11)
        for point in points:
            self.cell(10, 6, "-", ln=0)
            wrapped = textwrap.wrap(point, 75)
            if wrapped:
                self.cell(0, 6, wrapped[0], ln=True)
                for line in wrapped[1:]:
                    self.cell(10, 6, "", ln=0)
                    self.cell(0, 6, line, ln=True)
            else:
                self.cell(0, 6, "", ln=True)
        self.ln(5)

    def add_image(self, img_path, w=180):
        self.image(img_path, x=15, y=None, w=w)
        self.ln(5)

    def image_caption(self, caption):
        self.set_font("Arial", "I", 9)
        self.cell(0, 6, caption, ln=True, align="C")
        self.set_font("Arial", "", 11)
        self.ln(10)

    def add_footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')} | {self.company_name} Confidential", 0, 0, "C")

# Upload Endpoint
@app.post("/upload/", response_model=dict)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a data file"""
    session_id = str(uuid.uuid4())
    file_location = os.path.join("/tmp", f"{session_id}_{file.filename}")
    
    with open(file_location, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_location)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_location)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file_location)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        data_preview = df.head(5).to_dict(orient="records")
        columns = df.columns.tolist()
        row_count = len(df)

        session_data = {
            "file_path": file_location,
            "columns": columns,
            "row_count": row_count,
            "created_at": datetime.datetime.now().isoformat()
        }

        with open(os.path.join("/tmp", f"{session_id}_info.json"), "w") as f:
            json.dump(session_data, f, cls=NumpyEncoder)

        return {
            "status": "success",
            "session_id": session_id,
            "file_name": file.filename,
            "columns": columns,
            "row_count": row_count,
            "preview": data_preview
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Analyze Endpoint
@app.post("/analyze/{session_id}", response_model=dict)
async def analyze_data(session_id: str):
    """Perform basic analysis"""
    try:
        with open(os.path.join("/tmp", f"{session_id}_info.json"), "r") as f:
            session_data = json.load(f)

        file_path = session_data["file_path"]

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)

        analyzer = CorporateDataAnalyzer(df)
        insights = analyzer.generate_business_insights()
        visualizations = analyzer.create_corporate_visualizations(session_id=session_id)
        recommendations = analyzer.recommendations

        session_data.update({
            "analysis_complete": True,
            "insights": insights,
            "visualizations": visualizations,
            "recommendations": recommendations
        })

        with open(os.path.join("/tmp", f"{session_id}_info.json"), "w") as f:
            json.dump(session_data, f, cls=NumpyEncoder)

        return {
            "status": "success",
            "insights": insights,
            "visualizations": visualizations,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")

# Generate Story Endpoint
@app.post("/generate-story/{session_id}", response_model=StoryResponse)
async def generate_story(session_id: str, request: StoryGenerationRequest):
    """Generate business data story"""
    try:
        with open(os.path.join("/tmp", f"{session_id}_info.json"), "r") as f:
            session_data = json.load(f)

        if not session_data.get("analysis_complete", False):
            raise HTTPException(status_code=400, detail="Analysis not complete.")

        insights = session_data.get("insights", [])
        visualizations = session_data.get("visualizations", [])
        recommendations = session_data.get("recommendations", [])

        report_content = await generate_llm_insights(
            insights, 
            request.business_context, 
            request.corporate_style, 
            request.audience
        )

        title = request.title or "Business Data Report"

        pdf_path = os.path.join("/tmp", f"{session_id}_report.pdf")
        pdf = CorporateReportPDF(title=title)

        for section in report_content.split('\n\n'):
            if section.strip().startswith('#'):
                pdf.section_title(section.replace('#', '').strip())
            elif section.strip().startswith('*'):
                bullets = [line.strip().replace('*', '').strip() for line in section.strip().split('\n')]
                pdf.add_bullet_points(bullets)
            else:
                if section.strip():
                    pdf.section_body(section.strip())

        pdf.add_footer()
        pdf.output(pdf_path)

        return StoryResponse(
            report_id=session_id,
            title=title,
            preview=report_content[:500] + "...",
            pdf_url=f"/download/{session_id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")

# Download PDF Endpoint
@app.get("/download/{session_id}", response_class=FileResponse)
async def download_report(session_id: str):
    """Download the generated PDF"""
    pdf_path = os.path.join("/tmp", f"{session_id}_report.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(pdf_path, filename="business_data_report.pdf", media_type="application/pdf")

# Root Page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
