import os
import torch
import torchaudio
import numpy as np
import librosa
import logging
from models.text_to_music import TextToMusicGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioToMusicGenerator:
    def __init__(self):
        """
        Initialize the audio-to-music generator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Initialize text-to-music generator
            self.text_to_music = TextToMusicGenerator(model_size="small")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _extract_melody_features(self, audio_path):
        """
        Extract melody features from the input audio.
        
        Args:
            audio_path (str): Path to the input audio file
            
        Returns:
            dict: Dictionary containing melody features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the pitch with highest magnitude at each time step
            pitch_indices = np.argmax(magnitudes, axis=0)
            pitches = np.array([pitches[pitch_idx, t] for t, pitch_idx in enumerate(pitch_indices)])
            
            # Filter out zero pitches (silence)
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) == 0:
                avg_pitch = 0
                pitch_range = 0
            else:
                avg_pitch = np.mean(valid_pitches)
                pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
            
            # Extract tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Extract rhythm features
            # Compute onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Detect onsets
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            
            # Calculate rhythm regularity (variance of inter-onset intervals)
            if len(onsets) > 1:
                onset_times = librosa.frames_to_time(onsets, sr=sr)
                ioi = np.diff(onset_times)
                rhythm_regularity = 1.0 / (1.0 + np.std(ioi))  # Normalize to 0-1 range
            else:
                rhythm_regularity = 0.5  # Default value if not enough onsets
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            # Determine if the melody is more "major" or "minor" based on chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Simplified major/minor detection based on relative presence of major/minor thirds
            major_third_idx = 4  # 4 semitones above root
            minor_third_idx = 3  # 3 semitones above root
            
            # Roll chroma to get all possible keys and check major/minor presence
            major_scores = []
            minor_scores = []
            
            for i in range(12):  # For all possible keys
                rolled_chroma = np.roll(chroma, i, axis=0)
                root_strength = np.mean(rolled_chroma[0])
                major_third_strength = np.mean(rolled_chroma[major_third_idx])
                minor_third_strength = np.mean(rolled_chroma[minor_third_idx])
                
                major_scores.append(root_strength + major_third_strength)
                minor_scores.append(root_strength + minor_third_strength)
            
            is_major = np.max(major_scores) > np.max(minor_scores)
            
            return {
                "average_pitch": float(avg_pitch),
                "pitch_range": float(pitch_range),
                "tempo": float(tempo),
                "rhythm_regularity": float(rhythm_regularity),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "is_major": bool(is_major)
            }
            
        except Exception as e:
            logger.error(f"Error extracting melody features: {e}")
            return {
                "average_pitch": 220.0,  # Default A3
                "pitch_range": 100.0,
                "tempo": 90.0,
                "rhythm_regularity": 0.5,
                "spectral_centroid": 1000.0,
                "spectral_bandwidth": 1000.0,
                "is_major": True
            }
    
    def _create_music_prompt_from_features(self, features):
        """
        Create a text prompt for music generation based on extracted audio features.
        
        Args:
            features (dict): Dictionary containing melody features
            
        Returns:
            str: Text prompt for music generation
        """
        # Determine pitch range description
        if features["pitch_range"] < 50:
            pitch_range_desc = "narrow pitch range"
        elif features["pitch_range"] < 200:
            pitch_range_desc = "moderate pitch range"
        else:
            pitch_range_desc = "wide pitch range"
        
        # Determine if the melody is high, medium, or low pitched
        if features["average_pitch"] < 150:
            pitch_desc = "low-pitched"
        elif features["average_pitch"] < 350:
            pitch_desc = "medium-pitched"
        else:
            pitch_desc = "high-pitched"
        
        # Determine rhythm description
        if features["rhythm_regularity"] < 0.3:
            rhythm_desc = "irregular rhythm"
        elif features["rhythm_regularity"] < 0.7:
            rhythm_desc = "moderately regular rhythm"
        else:
            rhythm_desc = "steady rhythm"
        
        # Determine timbre description based on spectral features
        if features["spectral_centroid"] < 800:
            timbre_desc = "warm, rich"
        elif features["spectral_centroid"] < 1500:
            timbre_desc = "balanced"
        else:
            timbre_desc = "bright, sharp"
        
        # Determine tonality
        tonality = "major key" if features["is_major"] else "minor key"
        
        # Create the prompt
        prompt = f"A {pitch_desc} melody with {pitch_range_desc} and {rhythm_desc}, in a {tonality} with {timbre_desc} tones"
        
        return prompt
    
    def generate(self, audio_path, output_path, duration=10.0, genre=None, tempo_adjustment=None, complexity=None):
        """
        Generate music based on input audio and optional parameters.
        
        Args:
            audio_path (str): Path to the input audio file
            output_path (str): Path where the generated audio will be saved
            duration (float): Duration of the generated audio in seconds
            genre (str, optional): Music genre to influence generation
            tempo_adjustment (int, optional): Percentage adjustment to the detected tempo (-50 to +50)
            complexity (int, optional): Arrangement complexity (0-100)
            
        Returns:
            dict: Dictionary containing generation results and analysis
        """
        try:
            # Extract melody features
            features = self._extract_melody_features(audio_path)
            logger.info(f"Extracted melody features: {features}")
            
            # Create text prompt from features
            prompt = self._create_music_prompt_from_features(features)
            logger.info(f"Generated prompt: {prompt}")
            
            # Adjust tempo if requested
            final_tempo = features["tempo"]
            if tempo_adjustment is not None:
                adjustment_factor = 1.0 + (float(tempo_adjustment) / 100.0)
                final_tempo = features["tempo"] * adjustment_factor
                final_tempo = max(40, min(200, final_tempo))  # Clamp to reasonable range
            
            # Adjust prompt based on complexity
            if complexity is not None:
                complexity_value = int(complexity)
                if complexity_value < 30:
                    prompt += " with minimal instrumentation, simple arrangement"
                elif complexity_value < 70:
                    prompt += " with moderate instrumentation"
                else:
                    prompt += " with rich instrumentation, complex arrangement"
            
            # Generate music using the text-to-music generator
            self.text_to_music.generate(
                text_prompt=prompt,
                output_path=output_path,
                duration=duration,
                genre=genre,
                tempo=final_tempo,
                mood=None  # Mood is already captured in the prompt
            )
            
            return {
                "output_path": output_path,
                "melody_features": features,
                "generated_prompt": prompt,
                "final_parameters": {
                    "tempo": final_tempo,
                    "genre": genre,
                    "complexity": complexity
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating music from audio: {e}")
            raise
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


# For testing
if __name__ == "__main__":
    generator = AudioToMusicGenerator()
    output_file = "test_audio_output.wav"
    result = generator.generate(
        audio_path="test_audio.wav",
        output_path=output_file,
        duration=5.0
    )
    print(f"Generated audio saved to {output_file}")
    print(f"Audio analysis: {result}")
