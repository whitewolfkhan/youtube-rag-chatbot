from typing import List, Dict, Optional
from utils import extract_video_id, clean_text

from youtube_transcript_api import YouTubeTranscriptApi


class TranscriptExtractor:
    """
    Extract transcripts from YouTube videos.
    """
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the transcript extractor.
        
        Args:
            languages: List of language codes to try (default: ['en'])
        """
        self.languages = languages or ['en', 'en-US', 'en-GB']
        self.api = YouTubeTranscriptApi()
    
    def extract(self, url: str) -> Dict:
        """
        Extract transcript from a YouTube video URL.
        """
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            return {
                'success': False,
                'error': 'Invalid YouTube URL. Could not extract video ID.'
            }
        
        try:
            # Fetch transcript
            transcript_data = self.api.fetch(video_id)
            
            # Process transcript
            full_text = ""
            chunks = []
            
            for entry in transcript_data:
                text = clean_text(entry.text)
                start_time = entry.start
                duration = entry.duration
                
                full_text += text + " "
                
                chunks.append({
                    'text': text,
                    'start': start_time,
                    'end': start_time + duration,
                    'duration': duration
                })
            
            return {
                'success': True,
                'video_id': video_id,
                'text': full_text.strip(),
                'chunks': chunks,
                'language': 'en',
                'total_duration': chunks[-1]['end'] if chunks else 0
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'transcripts disabled' in error_msg or 'disabled' in error_msg:
                return {
                    'success': False,
                    'error': 'Transcripts are disabled for this video.'
                }
            elif 'video unavailable' in error_msg or 'private' in error_msg or 'not found' in error_msg:
                return {
                    'success': False,
                    'error': 'Video is unavailable or private.'
                }
            elif 'no transcript' in error_msg:
                return {
                    'success': False,
                    'error': 'No transcript found for this video. The video may not have captions.'
                }
            else:
                return {
                    'success': False,
                    'error': f'An error occurred: {str(e)}'
                }


# Create a default instance
transcript_extractor = TranscriptExtractor()
