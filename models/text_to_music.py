import os
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextToMusicGenerator:
    def __init__(self, model_size="small"):
        """
        Initialize the text-to-music generator with the specified model size.

        Args:
            model_size (str): Size of the MusicGen model to use. Options: "small", "medium", "large".
                              Smaller models are faster but may produce lower quality results.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Map model size to the corresponding model ID
        model_map = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium",
            "large": "facebook/musicgen-large"
        }

        model_id = model_map.get(model_size, "facebook/musicgen-small")

        try:
            logger.info(f"Loading MusicGen model: {model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)

            # Move model to the appropriate device
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate(self, text_prompt, output_path, duration=10.0, genre=None, tempo=None, mood=None):
        """
        Generate music based on a text prompt and optional parameters.

        Args:
            text_prompt (str): Textual description of the music to generate
            output_path (str): Path where the generated audio will be saved
            duration (float): Duration of the generated audio in seconds
            genre (str, optional): Music genre to influence generation
            tempo (int, optional): Tempo in BPM
            mood (int, optional): Mood value from 0 (sad) to 100 (happy)

        Returns:
            str: Path to the generated audio file
        """
        try:
            # Enhance the prompt with additional parameters if provided
            enhanced_prompt = text_prompt

            if genre:
                enhanced_prompt += f" in {genre} style"

            if tempo:
                tempo_description = "fast-paced" if int(tempo) > 120 else "medium-paced" if int(tempo) > 80 else "slow-paced"
                enhanced_prompt += f", {tempo_description}"

            if mood is not None:
                mood_value = int(mood)
                mood_description = "happy and uplifting" if mood_value > 70 else \
                                "positive and bright" if mood_value > 55 else \
                                "neutral" if mood_value > 45 else \
                                "melancholic" if mood_value > 30 else "sad and somber"
                enhanced_prompt += f", {mood_description}"

            logger.info(f"Enhanced prompt: {enhanced_prompt}")

            # Set generation parameters
            self.model.generation_config.max_new_tokens = int(duration * 50)  # Approximate tokens for duration

            # Generate the audio
            inputs = self.processor(
                text=[enhanced_prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logger.info("Generating audio...")
            audio_values = self.model.generate(**inputs, do_sample=True, guidance_scale=3.0)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Handle different tensor dimensions
            if len(audio_values.shape) == 3:
                # If we have a 3D tensor [batch, channels, time]
                audio_tensor = audio_values[0]  # Take the first batch
            else:
                # If we have a 2D tensor [channels, time]
                audio_tensor = audio_values

            # Convert to CPU for saving
            audio_tensor = audio_tensor.cpu()

            # Make sure it's 2D [channels, time] for torchaudio
            if len(audio_tensor.shape) == 1:
                # If it's a 1D tensor [time], add channel dimension
                audio_tensor = audio_tensor.unsqueeze(0)

            # Save as WAV file
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            torchaudio.save(
                output_path,
                audio_tensor,
                sample_rate=sampling_rate
            )

            logger.info(f"Audio saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating music: {e}")
            raise

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


# For testing
if __name__ == "__main__":
    generator = TextToMusicGenerator(model_size="small")
    output_file = "test_output.wav"
    generator.generate(
        text_prompt="A cheerful piano melody with light percussion",
        output_path=output_file,
        duration=5.0
    )
    print(f"Generated audio saved to {output_file}")
