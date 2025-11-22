"""
End-to-End Emotional Landscape Generator
Analyzes music emotion and generates corresponding landscape images.

Usage:
    python main.py --audio path/to/music.mp3 --output output_image.png
    python main.py --valence 7.5 --arousal 6.0 --output output_image.png
"""

import numpy as np
import librosa
import pandas as pd
from joblib import load
import os
import sys
import argparse

# Add Image_Generation/Diffusion to path for generate import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Image_Generation', 'Diffusion'))
from generate import generate_emotion_image

# Configuration - Update these paths as needed
MUSIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Weights", "Music", "music_model_optimized.joblib")
DIFFUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Weights", "Diffusion", "diffusion_epoch1650.pth")
DEFAULT_OUTPUT_PATH = "generated_landscape.png"


def extract_audio_features(music_file):
    """
    Extract audio features from music file for emotion prediction.
    
    Args:
        music_file: Path to audio file
    
    Returns:
        numpy array of audio features
    """
    # Load the audio file
    y, sr = librosa.load(music_file, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

    # Aggregation function
    def aggregate(feature_matrix):
        return np.concatenate(
            [np.mean(feature_matrix, axis=1), np.std(feature_matrix, axis=1)]
        )

    # Combine all features into a single array
    features = np.concatenate(
        [
            aggregate(mfccs),
            aggregate(chroma_stft),
            aggregate(spectral_contrast),
            aggregate(zero_crossing_rate),
            aggregate(np.array([tempo]).reshape(-1, 1)),
            aggregate(rms),
            aggregate(spectral_centroid),
            aggregate(spectral_bandwidth),
            aggregate(spectral_rolloff),
            aggregate(tonnetz),
            aggregate(chroma_cqt),
            aggregate(chroma_cens),
        ]
    )

    return features


def predict_music_emotion(audio_path, model_path=None):
    """
    Predict valence and arousal from audio file.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        model_path: Path to music model (optional, uses default if not provided)
    
    Returns:
        tuple: (valence, arousal) values on 1-9 scale
    """
    if model_path is None:
        model_path = MUSIC_MODEL_PATH
        
    # Load music model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Music model not found at: {model_path}\n"
                              f"Please ensure the model file exists or update MUSIC_MODEL_PATH in main.py")
    
    print(f"Loading music emotion model...")
    trained_model = load(model_path)
    print("Music model loaded successfully!")
    
    # Extract features from audio
    print(f"Analyzing music: {audio_path}")
    features = extract_audio_features(audio_path)
    
    # Build column names (must match training)
    columns = []
    for i in range(13):
        columns.append(f"mfcc_dct{i}_mean")
        columns.append(f"mfcc_dct{i}_std")
    for i in range(12):
        columns.append(f"chroma_stft_chord{i}_mean")
        columns.append(f"chroma_stft_chord{i}_std")
    for i in range(7):
        columns.append(f"spectral_contrast_frequency{i}_mean")
        columns.append(f"spectral_contrast_frequency{i}_std")
    for i in range(1):
        columns.append(f"zero_crossing_rate_frame{i}_mean")
        columns.append(f"zero_crossing_rate_frame{i}_std")
    columns.append("tempo_mean")
    columns.append("tempo_std")
    columns.append("rms_mean")
    columns.append("rms_std")
    columns.append("spectral_centroid_mean")
    columns.append("spectral_centroid_std")
    columns.append("spectral_bandwidth_mean")
    columns.append("spectral_bandwidth_std")
    columns.append("spectral_rolloff_mean")
    columns.append("spectral_rolloff_std")
    for i in range(6):
        columns.append(f"tonnetz_dim{i}_mean")
        columns.append(f"tonnetz_dim{i}_std")
    for i in range(12):
        columns.append(f"chroma_cqt_chord{i}_mean")
        columns.append(f"chroma_cqt_chord{i}_std")
    for i in range(12):
        columns.append(f"chroma_cens_chord{i}_mean")
        columns.append(f"chroma_cens_chord{i}_std")

    # Reshape and create DataFrame
    features_reshaped = features.reshape(1, -1)
    features_df = pd.DataFrame(features_reshaped, columns=columns)

    # Predict valence and arousal
    predicted_va = trained_model.predict(features_df)
    valence = float(predicted_va[0][0])
    arousal = float(predicted_va[0][1])

    print(f"✓ Predicted Valence: {valence:.2f}/9.0 (1=sad, 9=happy)")
    print(f"✓ Predicted Arousal: {arousal:.2f}/9.0 (1=calm, 9=energetic)")
    
    return valence, arousal


def main():
    """Main entry point for the emotional landscape generator"""
    parser = argparse.ArgumentParser(
        description='Generate emotional landscape images from music or VA values'
    )
    parser.add_argument('--audio', '-a', type=str, 
                       help='Path to audio file for emotion analysis')
    parser.add_argument('--valence', '-v', type=float,
                       help='Valence value (1-9, where 1=sad, 9=happy)')
    parser.add_argument('--arousal', '-ar', type=float,
                       help='Arousal value (1-9, where 1=calm, 9=energetic)')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_PATH,
                       help='Output image path (default: generated_landscape.png)')
    parser.add_argument('--model', '-m', type=str, default=DIFFUSION_MODEL_PATH,
                       help='Path to diffusion model checkpoint')
    parser.add_argument('--guidance', '-g', type=float, default=5.0,
                       help='Guidance scale for image generation (3-7 typical, default: 5.0)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducibility (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.audio and (args.valence or args.arousal):
        print("Error: Cannot specify both --audio and --valence/--arousal")
        return
    
    if not args.audio and not (args.valence and args.arousal):
        print("Error: Must specify either --audio OR both --valence and --arousal")
        print("\nExamples:")
        print("  python main.py --audio music.mp3 --output landscape.png")
        print("  python main.py --valence 7.5 --arousal 6.0 --output landscape.png")
        return
    
    # Step 1: Get valence and arousal values
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            return
        valence, arousal = predict_music_emotion(args.audio)
    else:
        valence = args.valence
        arousal = args.arousal
        print(f"Using provided VA values: Valence={valence:.2f}, Arousal={arousal:.2f}")
    
    # Validate VA values
    if not (1 <= valence <= 9) or not (1 <= arousal <= 9):
        print(f"Warning: VA values should be in range [1, 9]")
        print(f"Current: Valence={valence:.2f}, Arousal={arousal:.2f}")
    
    # Step 2: Generate landscape image
    print(f"\n{'='*60}")
    print("GENERATING EMOTIONAL LANDSCAPE IMAGE")
    print(f"{'='*60}")
    
    if not os.path.exists(args.model):
        print(f"Error: Diffusion model not found: {args.model}")
        print(f"Please ensure the checkpoint exists or update DIFFUSION_MODEL_PATH in main.py")
        return
    
    generate_emotion_image(
        v=valence,
        a=arousal,
        model_path=args.model,
        save_path=args.output,
        guidance_scale=args.guidance,
        seed=args.seed
    )
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! Generated image saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
