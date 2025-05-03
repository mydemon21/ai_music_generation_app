import os
import torch
import numpy as np
from PIL import Image
import logging
from transformers import CLIPProcessor, CLIPModel
from models.text_to_music import TextToMusicGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageToMusicGenerator:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the image-to-music generator.
        
        Args:
            clip_model_name (str): Name of the CLIP model to use for image understanding
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load CLIP model for image understanding
            logger.info(f"Loading CLIP model: {clip_model_name}")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            
            # Initialize text-to-music generator
            self.text_to_music = TextToMusicGenerator(model_size="small")
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _extract_image_features(self, image_path):
        """
        Extract features from the image using CLIP.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Image features
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            return image_features
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            raise
    
    def _generate_image_description(self, image_features):
        """
        Generate a textual description of the image based on its features.
        This is a simplified approach - in a production system, you might use
        a more sophisticated image captioning model.
        
        Args:
            image_features (torch.Tensor): Features extracted from the image
            
        Returns:
            str: Textual description of the image
        """
        # Define a set of candidate descriptions
        candidate_descriptions = [
            "a serene landscape with mountains and lakes",
            "a vibrant cityscape with tall buildings",
            "a peaceful beach scene with waves",
            "a dark and moody forest",
            "a bright and colorful abstract painting",
            "a portrait with emotional expression",
            "a lively party scene with people dancing",
            "a calm and minimalist composition",
            "a dramatic sunset over the horizon",
            "a cozy indoor scene with warm lighting"
        ]
        
        # Convert descriptions to CLIP text features
        text_inputs = self.clip_processor(text=candidate_descriptions, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get the most similar description
        values, indices = similarity[0].topk(3)
        
        # Combine top descriptions for a more nuanced prompt
        top_descriptions = [candidate_descriptions[idx] for idx in indices]
        
        # Create a combined description
        combined_description = f"Music inspired by {top_descriptions[0]}"
        if values[1] > 0.1:  # If second description is also relevant
            combined_description += f" with elements of {top_descriptions[1]}"
        
        return combined_description
    
    def _analyze_image_mood(self, image_path):
        """
        Analyze the mood of the image based on color distribution.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing mood analysis results
        """
        try:
            # Open the image and convert to RGB
            image = Image.open(image_path).convert("RGB")
            
            # Resize for faster processing
            image = image.resize((100, 100))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate average RGB values
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Calculate brightness (simple average of RGB)
            brightness = np.mean(avg_color)
            
            # Calculate saturation
            r, g, b = avg_color
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            saturation = 0 if max_rgb == 0 else (max_rgb - min_rgb) / max_rgb
            
            # Calculate color dominance
            color_names = ["red", "green", "blue"]
            dominant_color = color_names[np.argmax(avg_color)]
            
            # Determine mood based on color analysis
            # High brightness + high saturation = energetic
            # High brightness + low saturation = peaceful
            # Low brightness + high saturation = intense
            # Low brightness + low saturation = melancholic
            
            mood_score = 50  # Neutral starting point
            
            # Adjust based on brightness (0-100)
            mood_score += (brightness / 255) * 30 - 15  # -15 to +15
            
            # Adjust based on saturation
            if saturation > 0.5:
                mood_score += 10
            else:
                mood_score -= 10
                
            # Adjust based on dominant color
            if dominant_color == "red":
                mood_score += 10  # Red is energetic/passionate
            elif dominant_color == "blue":
                mood_score -= 10  # Blue is calm/melancholic
            elif dominant_color == "green":
                mood_score += 5   # Green is balanced/natural
            
            # Clamp to 0-100 range
            mood_score = max(0, min(100, mood_score))
            
            # Determine tempo based on brightness and saturation
            tempo = 80  # Default moderate tempo
            
            if brightness > 180 and saturation > 0.6:
                tempo = 120  # Bright and saturated = faster
            elif brightness < 80 and saturation < 0.3:
                tempo = 60   # Dark and desaturated = slower
            
            return {
                "mood_score": int(mood_score),
                "brightness": int(brightness),
                "saturation": float(saturation),
                "dominant_color": dominant_color,
                "suggested_tempo": tempo
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image mood: {e}")
            return {
                "mood_score": 50,
                "brightness": 128,
                "saturation": 0.5,
                "dominant_color": "neutral",
                "suggested_tempo": 90
            }
    
    def generate(self, image_path, output_path, duration=10.0, genre=None, tempo=None, mood_influence=None):
        """
        Generate music based on an image and optional parameters.
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path where the generated audio will be saved
            duration (float): Duration of the generated audio in seconds
            genre (str, optional): Music genre to influence generation
            tempo (int, optional): Tempo in BPM (overrides image-based tempo if provided)
            mood_influence (int, optional): How strongly the image mood affects generation (0-100)
            
        Returns:
            dict: Dictionary containing generation results and analysis
        """
        try:
            # Extract image features
            image_features = self._extract_image_features(image_path)
            
            # Generate image description
            image_description = self._generate_image_description(image_features)
            logger.info(f"Generated image description: {image_description}")
            
            # Analyze image mood
            mood_analysis = self._analyze_image_mood(image_path)
            logger.info(f"Image mood analysis: {mood_analysis}")
            
            # Determine final parameters
            final_tempo = tempo if tempo is not None else mood_analysis["suggested_tempo"]
            
            # Determine how much the image mood affects the generation
            mood_weight = 1.0
            if mood_influence is not None:
                mood_weight = float(mood_influence) / 50.0  # 0-100 scale to 0-2 scale
            
            # Apply mood weight to the mood score
            weighted_mood = 50 + (mood_analysis["mood_score"] - 50) * mood_weight
            weighted_mood = max(0, min(100, weighted_mood))
            
            # Generate music using the text-to-music generator
            self.text_to_music.generate(
                text_prompt=image_description,
                output_path=output_path,
                duration=duration,
                genre=genre,
                tempo=final_tempo,
                mood=weighted_mood
            )
            
            return {
                "output_path": output_path,
                "image_description": image_description,
                "mood_analysis": mood_analysis,
                "final_parameters": {
                    "tempo": final_tempo,
                    "mood": weighted_mood,
                    "genre": genre
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating music from image: {e}")
            raise
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


# For testing
if __name__ == "__main__":
    generator = ImageToMusicGenerator()
    output_file = "test_image_output.wav"
    result = generator.generate(
        image_path="test_image.jpg",
        output_path=output_file,
        duration=5.0
    )
    print(f"Generated audio saved to {output_file}")
    print(f"Image analysis: {result}")
