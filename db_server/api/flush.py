import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = os.environ.get("DB_MEDIA_ROOT", "media/archive")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/flush")
async def flush_event(
    event_id: str = Form(...),
    metadata: str = Form(...),
    video_clip: Optional[UploadFile] = File(None)
):
    """
    Receives JSON metadata and an optional MP4 video clip 
    from the model_server's flush_worker.
    """
    logger.info(f"Received flush request for event: {event_id}")
    
    # Process metadata (JSON payload)
    try:
        # In a real app, parse the JSON and save to DB
        # data = json.loads(metadata)
        # EpisodeReview.objects.create(...)
        pass
    except Exception as e:
        logger.error(f"Failed parsing metadata: {e}")
        raise HTTPException(status_code=400, detail="Invalid metadata")

    # Save video clip if provided
    if video_clip:
        file_ext = os.path.splitext(video_clip.filename)[1]
        save_path = os.path.join(UPLOAD_DIR, f"{event_id}{file_ext}")
        
        try:
            with open(save_path, "wb") as buffer:
                content = await video_clip.read()
                buffer.write(content)
            logger.info(f"Saved video clip to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save video clip: {e}")
            raise HTTPException(status_code=500, detail="Failed saving clip")
            
    return {"status": "success", "event_id": event_id}
