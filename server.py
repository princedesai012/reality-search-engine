import os
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ai_processor import AISceneAnalyzer
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Reality Search Engine")

# Initialize AI analyzer
analyzer = AISceneAnalyzer()

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="frames_db", html=True), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Update the search endpoint in server.py
@app.post("/api/search")
async def search(query: str = Form(...)):
    try:
        results = analyzer.search_memory(query)
        # Convert results to a serializable format
        serialized_results = []
        for result in results:
            metadata = result['metadata']
            # Convert Windows path to web URL
            filepath = metadata.get('filepath', '').replace('\\', '/')
            filename = os.path.basename(filepath)
            
            serialized_results.append({
                'id': result['id'],
                'similarity': result['similarity'],
                'image_url': f"/static/{filename}",
                'label': metadata.get('top_label', ''),
                'confidence': metadata.get('confidence', 0),
                'timestamp': metadata.get('timestamp', '')
            })
        return {"results": serialized_results}
    except Exception as e:
        print(f"Search error: {str(e)}")  # Add error logging
        return {"error": str(e), "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
