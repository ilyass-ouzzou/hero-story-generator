import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# DB
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# AI / TTS
from google import genai
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------
# CONFIG / PATHS
# -------------------------
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
IMG_DIR = STATIC_DIR / "images"
AUD_DIR = STATIC_DIR / "audio"

# Create directories
for d in [STATIC_DIR, IMG_DIR, AUD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_URL = f"sqlite:///{(BASE_DIR / 'stories.db').as_posix()}"

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env file")
if not ELEVENLABS_API_KEY:
    logger.warning("Missing ELEVENLABS_API_KEY - audio generation will be disabled")

# -------------------------
# APP + CORS + STATIC
# -------------------------
app = FastAPI(title="Hero Story Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -------------------------
# DB SETUP
# -------------------------
Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Story(Base):
    __tablename__ = "stories"
    id = Column(Integer, primary_key=True, index=True)
    hero_description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    hero_image_path = Column(Text, nullable=True)
    outline_raw = Column(Text, nullable=True)
    scenes = relationship("Scene", back_populates="story", cascade="all, delete-orphan")

class Scene(Base):
    __tablename__ = "scenes"
    id = Column(Integer, primary_key=True, index=True)
    story_id = Column(Integer, ForeignKey("stories.id", ondelete="CASCADE"))
    number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    narration = Column(Text, nullable=False)
    image_path = Column(Text, nullable=True)
    audio_path = Column(Text, nullable=True)

    # NEW:
    image_status = Column(String(20), nullable=False, default="pending")  # pending / generating / ready / failed
    audio_status = Column(String(20), nullable=False, default="pending")

    story = relationship("Story", back_populates="scenes")

# Create tables
Base.metadata.create_all(bind=engine)

# -------------------------
# AI CLIENTS
# -------------------------
try:
    gclient = genai.Client(api_key=GOOGLE_API_KEY)
    logger.info("Google AI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google AI client: {e}")
    raise

try:
    eclient = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
    if eclient:
        logger.info("ElevenLabs client initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize ElevenLabs client: {e}")
    eclient = None

CHAT_SESSIONS: Dict[int, any] = {}

# -------------------------
# DEPENDENCY INJECTION
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# SCHEMAS
# -------------------------
class HeroInput(BaseModel):
    heroDescription: str  # Match frontend field name

class SceneResponse(BaseModel):
    id: int
    number: int
    title: str
    description: str
    narration: str
    image: Optional[str] = None  # Match frontend field name
    audio_url: Optional[str] = None

class StoryResponse(BaseModel):
    story_id: int
    hero_description: str
    hero_image_url: Optional[str] = None
    scenes: List[SceneResponse]

class SceneMediaResponse(BaseModel):
    image_url: Optional[str] = None
    audio_url: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def url_for_static(path: Optional[str]) -> Optional[str]:
    """Convert file path to static URL"""
    if not path:
        return None
    try:
        rel_path = Path(path).relative_to(STATIC_DIR)
        return f"/static/{rel_path.as_posix()}"
    except ValueError:
        logger.error(f"Path {path} is not relative to static directory")
        return None

def safe_filename(prefix: str, ext: str) -> str:
    """Generate a safe, unique filename"""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"

def generate_scene_media_task(story_id: int, scene_number: int):
    """Background worker: generate image and audio for one scene and update DB."""
    db = SessionLocal()
    try:
        story = db.query(Story).filter_by(id=story_id).first()
        if not story:
            logger.error(f"[BG] Story {story_id} not found")
            return

        scene = db.query(Scene).filter_by(story_id=story_id, number=scene_number).first()
        if not scene:
            logger.error(f"[BG] Scene {scene_number} for story {story_id} not found")
            return

        # --- IMAGE ---
        try:
            scene.image_status = "generating"
            db.commit()
            chat = ensure_chat_for_story(story_id)

            image_prompt = f""" Generate a cinematic scene image for: Scene {scene.number}: {scene.title}
Description: {scene.description}
Hero: {story.hero_description}
Style: Cinematic, movie-quality, dramatic lighting
Ensure the hero's appearance is consistent with previous images. Make it visually compelling and story-appropriate. """
            response = chat.send_message(image_prompt)

            image_data = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    break

            if image_data:
                filename = safe_filename(f"story{story_id}_scene{scene.number}", "png")
                filepath = IMG_DIR / filename
                with open(filepath, "wb") as f:
                    f.write(image_data)
                scene.image_path = str(filepath)
                scene.image_status = "ready"
                db.commit()
                logger.info(f"[BG] Scene image saved for story {story_id}, scene {scene.number}")
            else:
                scene.image_status = "failed"
                db.commit()
                logger.warning(f"[BG] No image returned for story {story_id}, scene {scene.number}")
        except Exception as e:
            logger.exception(f"[BG] Image generation failed for story {story_id} scene {scene.number}: {e}")
            scene.image_status = "failed"
            db.commit()

        # --- AUDIO ---
        if eclient and scene.narration:
            try:
                scene.audio_status = "generating"
                db.commit()
                
                # Generate audio - this returns a generator
                audio_generator = eclient.text_to_speech.convert(
                    text=scene.narration,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                
                # Create file path
                filename = safe_filename(f"story{story_id}_scene{scene.number}", "mp3")
                filepath = AUD_DIR / filename
                
                # Write audio chunks to file
                chunk_count = 0
                total_bytes = 0
                
                with open(filepath, "wb") as f:
                    for chunk in audio_generator:
                        if chunk and len(chunk) > 0:
                            f.write(chunk)
                            chunk_count += 1
                            total_bytes += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Verify file was created
                if filepath.exists() and filepath.stat().st_size > 0:
                    scene.audio_path = str(filepath)
                    scene.audio_status = "ready"
                    db.commit()
                    logger.info(f"[BG] Scene audio saved for story {story_id}, scene {scene.number}: {total_bytes} bytes")
                else:
                    scene.audio_status = "failed"
                    db.commit()
                    logger.error(f"[BG] Audio file is empty or doesn't exist")
                    
            except Exception as e:
                logger.exception(f"[BG] Audio generation failed for story {story_id} scene {scene.number}: {e}")
                scene.audio_status = "failed"
                db.commit()
        else:
            # If no eclient configured, mark audio as 'disabled'
            scene.audio_status = "disabled" if not eclient else scene.audio_status
            db.commit()

    finally:
        db.close()


def parse_outline_to_scenes(text: str) -> List[dict]:
    """Parse AI-generated outline into structured scenes"""
    scenes = []
    current_scene = {}
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        line_lower = line.lower()
        
        if line_lower.startswith('scene '):
            if current_scene:
                scenes.append(current_scene)
            # Extract scene number and title
            if ':' in line:
                parts = line.split(':', 1)
                title = parts[1].strip()
            else:
                title = f"Scene {len(scenes) + 1}"
            
            current_scene = {
                'number': len(scenes) + 1,
                'title': title,
                'description': '',
                'narration': ''
            }
        elif line_lower.startswith('description:'):
            if current_scene:
                current_scene['description'] = line.split(':', 1)[1].strip()
        elif line_lower.startswith('narration:'):
            if current_scene:
                current_scene['narration'] = line.split(':', 1)[1].strip()
        elif current_scene and not current_scene.get('description'):
            # If no explicit description, use the line as description
            current_scene['description'] = line
    
    # Add the last scene
    if current_scene:
        scenes.append(current_scene)
    
    # Ensure we have exactly 10 scenes
    while len(scenes) < 10:
        scene_num = len(scenes) + 1
        scenes.append({
            'number': scene_num,
            'title': f"Scene {scene_num}",
            'description': f"An exciting moment in the hero's journey.",
            'narration': f"The adventure continues as our hero faces new challenges."
        })
    
    return scenes[:10]

def ensure_chat_for_story(story_id: int):
    """Ensure a consistent chat session exists for the story"""
    if story_id not in CHAT_SESSIONS:
        try:
            CHAT_SESSIONS[story_id] = gclient.chats.create(model="gemini-2.5-flash-image-preview")
            logger.info(f"Created new chat session for story {story_id}")
        except Exception as e:
            logger.error(f"Failed to create chat session for story {story_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize AI chat session")
    
    return CHAT_SESSIONS[story_id]

# -------------------------
# API ENDPOINTS
# -------------------------
@app.get("/")
def root():
    return {"message": "Hero Story Generator API", "version": "1.0.0"}

@app.post("/api/generate-story", response_model=StoryResponse)
def create_story(data: HeroInput, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Generate a new hero story with 10 scenes"""
    try:
        logger.info(f"Creating story for hero: {data.heroDescription}")
        
        # Enhanced prompt for better story generation
        prompt = f"""
Create an epic 10-scene hero story. Be creative and engaging!

Hero Description: {data.heroDescription}

Format each scene EXACTLY like this:
Scene X: [Compelling Title]
Description: [Detailed visual scene description in 1-2 sentences]
Narration: [Engaging 1-2 sentence narration that moves the story forward]

Create scenes that follow a classic hero's journey:
1. The Ordinary World / Call to Adventure
2. Refusing the Call / Meeting the Mentor
3. Crossing the Threshold
4. Tests and Trials
5. Approaching the Ordeal
6. The Ordeal
7. The Reward
8. The Road Back
9. Resurrection/Final Challenge
10. Return with Elixir

Keep the hero visually and personality-wise consistent throughout all scenes.
Make it cinematic and exciting!
"""
        
        # Generate story outline using Google AI
        try:
            response = gclient.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[prompt]
            )
            outline_text = response.text or ""
            logger.info("Story outline generated successfully")
        except Exception as e:
            logger.error(f"Google AI generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")
        
        # Parse the outline into structured scenes
        scenes_data = parse_outline_to_scenes(outline_text)
        logger.info(f"Parsed {len(scenes_data)} scenes from outline")
        
        # Save to database
        try:
            story = Story(
                hero_description=data.heroDescription,
                outline_raw=outline_text
            )
            db.add(story)
            db.commit()
            db.refresh(story)
            
            # Create scene records
            db_scenes = []
            for scene_data in scenes_data:
                scene = Scene(
                    story_id=story.id,
                    number=scene_data['number'],
                    title=scene_data['title'],
                    description=scene_data['description'],
                    narration=scene_data['narration']
                )
                db.add(scene)
                db_scenes.append(scene)
            
            db.commit()
            
            # Refresh all scenes to get IDs
            for scene in db_scenes:
                db.refresh(scene)
            for scene in db_scenes:
                # add background task for each scene — starts immediately but runs async
                background_tasks.add_task(generate_scene_media_task, story.id, scene.number)
            logger.info(f"Background media generation scheduled for story {story.id}")
            
            logger.info(f"Story {story.id} created successfully with {len(db_scenes)} scenes")
            
            # Prepare response
            scenes_response = [
                SceneResponse(
                    id=scene.id,
                    number=scene.number,
                    title=scene.title,
                    description=scene.description,
                    narration=scene.narration,
                    image=None,
                    audio_url=None
                )
                for scene in db_scenes
            ]
            
            return StoryResponse(
                story_id=story.id,
                hero_description=story.hero_description,
                hero_image_url=None,
                scenes=scenes_response
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database error while creating story: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_story: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/stories/{story_id}/scenes/{scene_number}/status")
def scene_status(story_id: int, scene_number: int, db: Session = Depends(get_db)):
    """Return image/audio URLs and status for a scene"""
    story = db.query(Story).filter_by(id=story_id).first()
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    scene = db.query(Scene).filter_by(story_id=story_id, number=scene_number).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return {
        "image_url": url_for_static(scene.image_path),
        "audio_url": url_for_static(scene.audio_path),
        "image_status": scene.image_status,
        "audio_status": scene.audio_status
    }

@app.post("/api/stories/{story_id}/hero-image")
def generate_hero_image(story_id: int, db: Session = Depends(get_db)):
    """Generate a hero portrait image"""
    try:
        story = db.query(Story).filter_by(id=story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        chat = ensure_chat_for_story(story_id)
        
        prompt = f"""
Generate a detailed hero portrait image based on this description:
{story.hero_description}

Style: Cinematic, high-quality, professional artwork
Composition: Full-body or upper-body portrait
Background: Neutral or thematically appropriate
Lighting: Dramatic and heroic
Quality: High detail and resolution
"""
        
        try:
            response = chat.send_message(prompt)
            
            # Extract image from response
            image_data = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    break
            
            if not image_data:
                raise HTTPException(status_code=500, detail="No image generated by AI")
            
            # Save image
            filename = safe_filename(f"hero_{story_id}", "png")
            filepath = IMG_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            # Update database
            story.hero_image_path = str(filepath)
            db.commit()
            
            image_url = url_for_static(str(filepath))
            logger.info(f"Hero image generated for story {story_id}")
            
            return {"hero_image_url": image_url}
            
        except Exception as e:
            logger.error(f"Image generation failed for story {story_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_hero_image: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/stories/{story_id}/scenes/{scene_number}/generate")
def generate_scene_media(story_id: int, scene_number: int, db: Session = Depends(get_db)):
    """Generate image and audio for a specific scene"""
    try:
        story = db.query(Story).filter_by(id=story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        scene = db.query(Scene).filter_by(story_id=story_id, number=scene_number).first()
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        
        image_url = None
        audio_url = None
        
        # Generate scene image
        try:
            chat = ensure_chat_for_story(story_id)
            
            image_prompt = f"""
Generate a cinematic scene image for:
Scene {scene.number}: {scene.title}
Description: {scene.description}

Hero: {story.hero_description}

Style: Cinematic, movie-quality, dramatic lighting
Ensure the hero's appearance is consistent with previous images.
Make it visually compelling and story-appropriate.
"""
            
            response = chat.send_message(image_prompt)
            
            # Extract image
            image_data = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    break
            
            if image_data:
                filename = safe_filename(f"story{story_id}_scene{scene.number}", "png")
                filepath = IMG_DIR / filename
                
                with open(filepath, "wb") as f:
                    f.write(image_data)
                
                scene.image_path = str(filepath)
                image_url = url_for_static(str(filepath))
                logger.info(f"Scene image generated for story {story_id}, scene {scene_number}")
        
        except Exception as e:
            logger.warning(f"Scene image generation failed: {e}")
        
        # Generate scene audio (if ElevenLabs is available)
        if eclient and scene.narration:
            try:
                # Generate audio - this returns a generator
                audio_generator = eclient.text_to_speech.convert(
                    text=scene.narration,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",  # Professional narrator voice
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                
                # Create file path
                filename = safe_filename(f"story{story_id}_scene{scene.number}", "mp3")
                filepath = AUD_DIR / filename
                
                # Write audio chunks to file
                chunk_count = 0
                total_bytes = 0
                
                with open(filepath, "wb") as f:
                    for chunk in audio_generator:
                        if chunk and len(chunk) > 0:
                            f.write(chunk)
                            chunk_count += 1
                            total_bytes += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Verify file was created and update scene
                if filepath.exists() and filepath.stat().st_size > 0:
                    scene.audio_path = str(filepath)
                    audio_url = url_for_static(str(filepath))
                    logger.info(f"Scene audio generated for story {story_id}, scene {scene_number}: {total_bytes} bytes")
                else:
                    logger.error(f"Audio file is empty or doesn't exist")
                
            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")
                
        # Save updates to database
        db.commit()
        
        return SceneMediaResponse(
            image_url=image_url,
            audio_url=audio_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_scene_media: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/stories/{story_id}", response_model=StoryResponse)
def get_story(story_id: int, db: Session = Depends(get_db)):
    """Retrieve a complete story with all scenes"""
    try:
        story = db.query(Story).filter_by(id=story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        scenes = db.query(Scene).filter_by(story_id=story_id).order_by(Scene.number).all()
        
        scenes_response = [
            SceneResponse(
                id=scene.id,
                number=scene.number,
                title=scene.title,
                description=scene.description,
                narration=scene.narration,
                image=url_for_static(scene.image_path),
                audio_url=url_for_static(scene.audio_path)
            )
            for scene in scenes
        ]
        
        return StoryResponse(
            story_id=story.id,
            hero_description=story.hero_description,
            hero_image_url=url_for_static(story.hero_image_path),
            scenes=scenes_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving story {story_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving story: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "ok",
            "google_ai": "ok" if gclient else "error",
            "elevenlabs": "ok" if eclient else "disabled"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)