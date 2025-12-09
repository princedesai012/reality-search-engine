import os
import time
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import asyncio
import cv2
from alert_system import send_alert

# Try to import face_recognition safely
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("\n⚠️ WARNING: 'face_recognition' library not found.")
    print("   System will run in SAFE MODE (Face identification disabled).")
    print("   To enable: Install Visual Studio C++ Build tools and pip install dlib\n")

# Load environment variables
load_dotenv()

class AISceneAnalyzer:
    def __init__(self, persist_directory: str = "memory_db"):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize face recognition only if available
        self.known_face_encodings = []
        self.known_face_names = []
        if FACE_REC_AVAILABLE:
            self.load_known_faces()

        # Initialize CLIP model and processor
        print("Loading AI Model (CLIP)...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="frame_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text prompts for analysis
        self.text_descriptions = [
            'a person', 
            'a cat', 
            'an empty room', 
            'a thief',
            'a dog',
            'a vehicle',
            'a package',
            'suspicious activity'
        ]
        
        # Alert system settings
        self.last_alert_time = None
        self.ALERT_COOLDOWN = 30  # seconds
        
        # Pre-process text features
        self._setup_text_features()

    def _setup_text_features(self):
        """Setup text features for comparison."""
        text_inputs = self.processor(
            text=self.text_descriptions, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def load_known_faces(self, known_faces_dir: str = "known_faces") -> None:
        """Load known faces from the specified directory."""
        if not FACE_REC_AVAILABLE:
            return

        known_faces_dir = Path(known_faces_dir)
        if not known_faces_dir.exists():
            print(f"Warning: Known faces directory '{known_faces_dir}' not found.")
            return
            
        print("Loading known faces...")
        for face_file in known_faces_dir.glob("*.jpg"):
            try:
                image = face_recognition.load_image_file(face_file)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(face_file.stem)
                    print(f"Loaded: {face_file.stem}")
            except Exception as e:
                print(f"Error processing {face_file}: {e}")

    def get_latest_frame_path(self, frames_dir='frames_db') -> str:
        """Get the path of the most recent frame."""
        if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
            return None
        # Only look for .jpg files
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        if not frame_files:
            return None
        frame_files.sort(key=lambda x: os.path.getmtime(os.path.join(frames_dir, x)), reverse=True)
        return os.path.join(frames_dir, frame_files[0])

    async def analyze_latest_frame(self, frames_dir='frames_db') -> Dict[str, Any]:
        """Analyze the most recent frame."""
        frame_path = self.get_latest_frame_path(frames_dir)
        if not frame_path:
            print("No new frames found to analyze")
            return None
            
        print(f"\nAnalyzing frame: {frame_path}")  # Debug print
        frame_id = os.path.splitext(os.path.basename(frame_path))[0]
        
        try:
            # Check if already processed
            existing = self.collection.get(ids=[frame_id])
            if existing and len(existing['ids']) > 0:
                print(f"Frame {frame_id} already processed, skipping...")
                return None

            # Process image
            image = Image.open(frame_path)
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get similarity scores
            similarity_scores = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores = similarity_scores[0].cpu().numpy()
            
            top_idx = np.argmax(scores)
            top_label = self.text_descriptions[top_idx]
            confidence = float(scores[top_idx])
            
            # Check for alerts
            await self._check_for_alerts(frame_path, top_label, confidence)
            
            # Store results
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'top_label': top_label,
                'confidence': confidence,
                'filepath': frame_path
            }
            
            self.collection.add(
                ids=[frame_id],
                embeddings=[image_features[0].cpu().numpy().tolist()],
                metadatas=[metadata]
            )
            
            print(f"Analysis complete - {top_label} ({confidence*100:.1f}%)")
            return {
                'id': frame_id,
                'label': top_label,
                'confidence': confidence,
                'filepath': frame_path
            }
            
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _is_known_face(self, frame_path: str) -> Tuple[bool, Optional[str]]:
        """Check if known face exists (Safeguarded)."""
        if not FACE_REC_AVAILABLE or not self.known_face_encodings:
            return False, None
            
        try:
            image = face_recognition.load_image_file(frame_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    return True, self.known_face_names[matches.index(True)]
            return False, None
        except Exception:
            return False, None

    async def _check_for_alerts(self, frame_path: str, label: str, confidence: float):
        print(f"Checking alerts - Label: {label}, Confidence: {confidence:.2f}")  # Debug print
    
        if label in ['a person', 'suspicious activity'] and confidence > 0.30:
            print(f"🚨 Potential alert triggered for {label} with {confidence*100:.1f}% confidence")
        
            is_known = False
            person_name = None
            if FACE_REC_AVAILABLE:
                is_known, person_name = self._is_known_face(frame_path)
            
            if is_known:
                print(f"👤 Known person detected: {person_name} (No Alert)")
                return
                
            current_time = datetime.now()
            if (self.last_alert_time is None or 
                (current_time - self.last_alert_time).total_seconds() >= self.ALERT_COOLDOWN):
                
                print(f"🚨 ALERT: {label} detected! Sending Telegram...")
                try:
                    # Make sure the file exists before trying to send it
                    if os.path.exists(frame_path):
                        print(f"File exists, size: {os.path.getsize(frame_path)} bytes")
                        await asyncio.to_thread(send_alert, frame_path, f"Unknown {label}", confidence)
                        self.last_alert_time = current_time
                        print("Alert sent successfully!")
                    else:
                        print(f"Error: File not found at {frame_path}")
                except Exception as e:
                    print(f"Alert failed with error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                seconds_left = int(self.ALERT_COOLDOWN - (current_time - self.last_alert_time).total_seconds())
                print(f"Alert on cooldown. Next alert available in {seconds_left} seconds")
        else:
            print(f"No alert - Condition not met (label: {label}, confidence: {confidence:.2f})")


    def search_memory(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                query_embedding = self.model.get_text_features(**inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            
            results = self.collection.query(
                query_embeddings=[query_embedding[0].cpu().numpy().tolist()],
                n_results=min(top_k, 10)
            )
            
            matches = []
            if results['ids']:
                for i, (id_, dist, meta) in enumerate(zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
                    matches.append({'id': id_, 'similarity': float(1 - dist), 'metadata': meta})
            return matches
        except Exception as e:
            print(f"Search error: {e}")
            return []

async def main():
    analyzer = AISceneAnalyzer()
    try:
        print("1. Analyze Live  2. Search  3. Exit")
        choice = input("Choice: ").strip()
        if choice == '1':
            print("Analyzing... (Press Ctrl+C to stop)")
            try:
                while True:
                    result = await analyzer.analyze_latest_frame()
                    if result:  # Only print if we got a result
                        print(f"Analyzed: {result.get('id')} - {result.get('label')} ({result.get('confidence', 0)*100:.1f}%)")
                    await asyncio.sleep(2)  # 2 second delay to prevent high CPU usage
            except KeyboardInterrupt:
                print("\nStopping analysis...")
        elif choice == '2':
            q = input("Search query: ")
            res = analyzer.search_memory(q)
            for r in res:
                print(f"Found: {r['metadata']['filepath']} ({r['similarity']:.2f})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")