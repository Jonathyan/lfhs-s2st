import os
import torch
import numpy as np
from pathlib import Path
import torchaudio
import time
import traceback
from typing import List, Tuple, Optional

# Voor chunk-verwerking
from pydub import AudioSegment
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech, SeamlessM4Tv2Model

class SeamlessS2ST:
    """
    Een klasse voor Nederlands naar Indonesisch spraak-naar-spraak vertaling
    met stemkloning gebruikmakend van SeamlessM4T v2.
    """
    
    def __init__(self, device="mps", dtype=torch.float32):
        """
        Initialiseer de S2ST-vertaler met modellen voor M1 Pro.
        
        Args:
            device: 'mps' voor M1 Mac (Metal Performance Shaders)
            dtype: datatype voor tensors (moet float32 zijn voor MPS)
        """
        self.device = device
        self.dtype = dtype
        print(f"Initializing SeamlessM4T v2 on {device} with {dtype}...")
        
        # Load SeamlessM4T v2 models and processor
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
            "facebook/seamless-m4t-v2-large"
        ).to(device)
        
        print("Models loaded successfully!")
        
        # Default segmentation parameters (kan aangepast worden)
        self.chunk_size_ms = 30000  # 30-seconden chunks
        self.overlap_ms = 2000      # 2-seconden overlap voor soepelere overgangen
        
    def segment_audio(self, audio_path: str) -> List[AudioSegment]:
        """
        Verdeelt lange audio in beheerbare segmenten met overlap.
        
        Args:
            audio_path: Pad naar het audiobestand
            
        Returns:
            List van audiosegmenten
        """
        print(f"Segmenting audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        
        # Bereken het aantal segmenten
        total_ms = len(audio)
        segments = []
        
        for start_ms in range(0, total_ms, self.chunk_size_ms - self.overlap_ms):
            end_ms = min(start_ms + self.chunk_size_ms, total_ms)
            segment = audio[start_ms:end_ms]
            segments.append(segment)
            
            # Voortgang tonen
            progress = min(100, (end_ms / total_ms) * 100)
            print(f"Segmentation: {progress:.1f}% complete", end="\r")
            
        print(f"\nSegmented audio into {len(segments)} chunks")
        return segments
    
    def process_segment(self, 
                       audio_segment: AudioSegment,
                       reference_speech: Optional[torch.Tensor] = None,
                       src_lang: str = "nld", 
                       tgt_lang: str = "ind") -> Tuple[torch.Tensor, float]:
        """
        Verwerk een audiosegment: spraak-naar-spraak vertaling met stemkloning.
        
        Args:
            audio_segment: Het te verwerken audiosegment
            reference_speech: Referentie-audio voor stemkloning (optioneel)
            src_lang: Brontaal (Nederlands = "nld")
            tgt_lang: Doeltaal (Indonesisch = "ind")
            
        Returns:
            Tuple van (vertaald_audio, processing_tijd)
        """
        start_time = time.time()
        
        # Controleer of het segment geluid bevat
        if len(audio_segment) == 0 or audio_segment.dBFS < -60:
            print("Warning: Audio segment is silent or empty. This may cause issues.")
        
        # Converteer AudioSegment naar tensor
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        if audio_segment.channels > 1:
            # Convert stereo to mono by averaging channels
            audio_array = audio_array.reshape(-1, audio_segment.channels).mean(axis=1)
        
        # Normaliseer tussen -1 en 1
        max_possible_value = float(2**(audio_segment.sample_width * 8 - 1))
        audio_array = audio_array / max_possible_value
        
        # Controleer op NaN/Inf waarden die problemen kunnen veroorzaken
        if np.isnan(audio_array).any() or np.isinf(audio_array).any():
            print("WARNING: Audio contains NaN or Inf values, replacing with zeros")
            audio_array = np.nan_to_num(audio_array)
        
        # Zet om naar torch tensor met EXPLICIET float32 (cruciaal voor MPS)
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        sample_rate = audio_segment.frame_rate
        
        print(f"Audio tensor ready: shape={audio_tensor.shape}, sr={sample_rate}, min={audio_tensor.min().item()}, max={audio_tensor.max().item()}")
        
        # Resample naar 16kHz als nodig
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Ensure correct dtype for MPS compatibility
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=sample_rate, 
                new_freq=16000
            ).to(dtype=torch.float32)
            audio_tensor = audio_tensor.contiguous()
            sample_rate = 16000
            
        print(f"After resampling: shape={audio_tensor.shape}, non-zero elements: {torch.count_nonzero(audio_tensor).item()}")
        
        # Debug output voor een kleine sample van de audio
        if len(audio_tensor) > 1600:  # Show first 0.1 sec at 16kHz if long enough
            print("First 0.1s audio values:", audio_tensor[:1600:160])  # Just 10 values
                
        # Zorg voor juiste dimensies voor de processor (verwacht [batch, time])
        if audio_tensor.dim() == 1:  # [time] -> [batch, time]
            audio_tensor = audio_tensor.unsqueeze(0)
            print(f"Added batch dimension: {audio_tensor.shape}")
        
        # Move tensor to device AFTER all processing
        audio_tensor = audio_tensor.to(self.device)
        
        # Extra check om te bevestigen dat audio valide is voor de processor
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

        # Controle op nullen en NaN
        non_zero = torch.count_nonzero(audio_tensor).item()
        has_nan = torch.isnan(audio_tensor).any().item()
        print(f"Audio pre-processor check: shape={audio_tensor.shape}, non-zero={non_zero}, has_nan={has_nan}")

        # Force een kopie op CPU voor diagnose
        audio_cpu = audio_tensor.cpu().detach().numpy()
        print(f"Audio stats: min={audio_cpu.min()}, max={audio_cpu.max()}, mean={audio_cpu.mean()}")

        if non_zero == 0:
            print("WARNING: Audio tensor contains only zeros!")
            # Vervang door een dummy tone om te testen of dat werkt
            freq = 440  # Hz
            sample_rate = 16000
            time_points = torch.arange(0, 3*sample_rate) / sample_rate
            audio_tensor = torch.sin(2 * np.pi * freq * time_points).unsqueeze(0).to(self.device)
            print(f"Replaced with dummy tone, new shape: {audio_tensor.shape}")
        
        # Zorg voor betere MPS compatibiliteit via correcte dtypes
        try:
            print(f"Calling processor with: audio shape={audio_tensor.shape}, src={src_lang}, tgt={tgt_lang}")
            raw_inputs = self.processor(
                audios=audio_tensor, 
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                return_tensors="pt"
            )
            
            print(f"Processor inputs created successfully with keys: {raw_inputs.keys()}")
            if 'audio_features' in raw_inputs:
                print(f"Audio features shape: {raw_inputs['audio_features'].shape}")
            
            # Convert inputs to correct dtype and device
            inputs = {k: v.to(dtype=self.dtype, device=self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in raw_inputs.items()}
            
            with torch.no_grad():
                # Generate met of zonder voice cloning
                if reference_speech is not None:
                    print("Using voice cloning for generation")
                    generated_speech = self.model.generate(
                        **inputs,
                        speaker_embedding=reference_speech,
                        return_intermediate_token_ids=True
                    )
                else:
                    print("Generating without voice cloning")
                    generated_speech = self.model.generate(
                        **inputs,
                        return_intermediate_token_ids=True
                    )
            
            # Extract speech output
            translated_speech = generated_speech.waveform[0].cpu()
            print(f"Translation successful: output shape={translated_speech.shape}")
            
        except Exception as e:
            print(f"Error during segment processing: {e}")
            traceback_str = traceback.format_exc()
            print(f"Detailed traceback:\n{traceback_str}")
            
            # Create a silent segment of 5 seconds as a fallback
            print("Returning silent audio as fallback")
            translated_speech = torch.zeros(16000 * 5, dtype=torch.float32)
        
        elapsed_time = time.time() - start_time
        return translated_speech, elapsed_time
    
    def extract_voice_embedding(self, reference_audio_path: str) -> torch.Tensor:
        """
        Extraheer stem-embedding voor voice cloning.
        
        Args:
            reference_audio_path: Pad naar referentie-audio voor stemkloning
            
        Returns:
            Stem-embedding tensor
        """
        print(f"Extracting voice embedding from: {reference_audio_path}")
        
        # Controleer of het bestand bestaat en toegankelijk is
        if not os.path.isfile(reference_audio_path):
            raise FileNotFoundError(f"Voice reference file not found: {reference_audio_path}")
            
        print(f"File exists and is accessible: {reference_audio_path}")
        print(f"File size: {os.path.getsize(reference_audio_path)} bytes")
        
        # Gebruik pydub om de audio te laden en te controleren
        try:
            audio_segment = AudioSegment.from_file(reference_audio_path)
            print(f"Successfully loaded with pydub: {len(audio_segment)/1000}s, {audio_segment.channels} channels, {audio_segment.frame_rate}Hz")
            
            # Converteer AudioSegment naar numpy array en dan naar torch tensor
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            if audio_segment.channels > 1:
                samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
            
            # Normaliseer tussen -1 en 1
            samples = samples / (2**(audio_segment.sample_width * 8 - 1))
            
            # Converteer naar torch tensor
            audio = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            sr = audio_segment.frame_rate
            
            print(f"Converted to tensor: shape={audio.shape}, dtype={audio.dtype}, sr={sr}")
        except Exception as e:
            print(f"Error loading with pydub: {e}")
            # Fallback naar torchaudio als backup
            try:
                audio, sr = torchaudio.load(reference_audio_path)
                print(f"Fallback: torchaudio loaded shape={audio.shape}, sr={sr}")
            except Exception as e:
                print(f"Critical error loading audio: {e}")
                raise
        
        # Zorg ervoor dat we float32 gebruiken (MPS vereiste)
        audio = audio.to(dtype=torch.float32)
        
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz")
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=sr, 
                new_freq=16000
            ).to(dtype=torch.float32)
        
        # Zorg voor mono audio
        if audio.shape[0] > 1:
            print(f"Converting from {audio.shape[0]} channels to mono")
            audio = audio.mean(dim=0, keepdim=True).to(dtype=torch.float32)
        
        # Zorg dat het niet te lang is (neem max 10 seconden)
        if audio.shape[1] > 160000:  # 10 sec at 16kHz
            print(f"Trimming audio from {audio.shape[1]/16000}s to 10s")
            audio = audio[:, :160000].contiguous()
        
        # Debug info
        print(f"Final audio tensor: shape={audio.shape}, dtype={audio.dtype}, min={audio.min()}, max={audio.max()}")
        
        # Controleer op NaN/Inf waarden die problemen kunnen veroorzaken
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            print("WARNING: Audio contains NaN or Inf values, replacing with zeros")
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Zorg dat de audio niet stil is
        if audio.abs().max() < 1e-6:
            print("WARNING: Audio is almost silent, this may cause voice embedding issues")
        
        audio = audio.to(self.device)
            
        # Extract speaker embedding
        with torch.no_grad():
            try:
                raw_inputs = self.processor(audio=audio, return_tensors="pt")
                print(f"Processor inputs created successfully with keys: {raw_inputs.keys()}")
                
                # Convert inputs to correct dtype and device
                inputs = {k: v.to(dtype=self.dtype, device=self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in raw_inputs.items()}
                
                speaker_embedding = self.model.extract_speaker_embedding(**inputs)
                print(f"Speaker embedding extracted successfully: shape={speaker_embedding.shape}")
                
                return speaker_embedding
            except Exception as e:
                print(f"Error in voice embedding extraction: {e}")
                # Als fallback, maak een dummy embedding (hiermee is stemkloning niet mogelijk maar voorkomt crashes)
                print("Using dummy speaker embedding as fallback - voice cloning will not work properly")
                return None
    
    def translate_long_audio(self, 
                           input_path: str, 
                           output_path: str,
                           reference_audio_path: Optional[str] = None,
                           src_lang: str = "nld", 
                           tgt_lang: str = "ind") -> str:
        """
        Vertaal een lang audiobestand van Nederlands naar Indonesisch.
        
        Args:
            input_path: Pad naar de bronopname (Nederlands)
            output_path: Pad om de vertaalde audio op te slaan
            reference_audio_path: Pad naar stemreferentie (optioneel)
            src_lang: Brontaal (Nederlands = "nld")
            tgt_lang: Doeltaal (Indonesisch = "ind")
            
        Returns:
            Pad naar het vertaalde audiobestand
        """
        start_time = time.time()
        print(f"Starting translation of {input_path} from {src_lang} to {tgt_lang}")
        
        # Controleer of inputbestand bestaat
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        print(f"Input file exists and is accessible: {input_path}")
        print(f"Input file size: {os.path.getsize(input_path)} bytes")
        
        # Extract voice embedding if provided
        speaker_embedding = None
        if reference_audio_path:
            try:
                speaker_embedding = self.extract_voice_embedding(reference_audio_path)
                if speaker_embedding is not None:
                    print("Voice embedding extracted successfully")
                else:
                    print("Warning: Voice embedding could not be extracted, continuing without voice cloning")
            except Exception as e:
                print(f"Error extracting voice embedding: {e}")
                print("Continuing without voice cloning...")
        
        # Segment audio into manageable chunks
        segments = self.segment_audio(input_path)
        total_segments = len(segments)
        
        if total_segments == 0:
            raise ValueError(f"No audio segments were extracted from {input_path}. The file may be empty or corrupted.")
        
        # Process each segment
        translated_segments = []
        for i, segment in enumerate(segments):
            print(f"Processing segment {i+1}/{total_segments}")
            try:
                translated_audio, proc_time = self.process_segment(
                    segment, speaker_embedding, src_lang, tgt_lang
                )
                translated_segments.append(translated_audio)
                print(f"Segment {i+1} processed in {proc_time:.2f}s")
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")
                print("Skipping this segment and continuing...")
                continue
        
        if not translated_segments:
            raise RuntimeError("No segments were successfully translated. Cannot create output file.")
        
        # Combine translated segments
        print("Combining translated segments...")
        combined_audio = torch.cat(translated_segments, dim=0)
        
        # Save the result
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        torchaudio.save(
            output_path, 
            combined_audio.unsqueeze(0), 
            sample_rate=16000,
            format="wav"
        )
        
        total_time = time.time() - start_time
        print(f"Translation completed in {total_time/60:.2f} minutes")
        print(f"Output saved to: {output_path}")
        
        return output_path


def main():
    """
    Hoofdfunctie om de translator uit te voeren
    """
    import argparse
    
    # Command line argumenten voor meer flexibiliteit
    parser = argparse.ArgumentParser(description='Vertaal Nederlandse audio naar Indonesisch met SeamlessM4T v2')
    parser.add_argument('--input', type=str, default="input/dutch_sermon.wav", 
                        help='Pad naar het bronautdiobestand (Nederlands)')
    parser.add_argument('--output', type=str, default="output/indonesian_sermon.wav",
                        help='Pad om het vertaalde audiobestand op te slaan')
    parser.add_argument('--reference', type=str, default="input/voice_reference.wav", 
                        help='Pad naar stemreferentie voor stemkloning (optioneel)')
    parser.add_argument('--src-lang', type=str, default="nld", 
                        help='Brontaal (default: "nld" voor Nederlands)')
    parser.add_argument('--tgt-lang', type=str, default="ind", 
                        help='Doeltaal (default: "ind" voor Indonesisch)')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device om te gebruiken (default: auto-detect)')
    parser.add_argument('--chunk-size', type=int, default=30000, 
                        help='Grootte van audiochunks in ms (default: 30000)')
    parser.add_argument('--test-mode', action='store_true', 
                        help='Start in testmodus met gegenereerde testbestanden')
    
    args = parser.parse_args()
    
    # Configureer de paden
    input_path = args.input
    output_path = args.output
    reference_voice_path = args.reference
    
    # Test mode - genereert en gebruikt testbestanden
    if args.test_mode:
        print("Starting in TEST MODE with generated test audio files")
        try:
            import subprocess
            
            # Maak input directory
            os.makedirs("input", exist_ok=True)
            
            # Test of we sox kunnen gebruiken
            try:
                subprocess.run(["sox", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                print("Using sox to generate test audio files")
                
                # Maak eenvoudige testbestanden met sox
                subprocess.run(["sox", "-n", "input/test_sermon.wav", "synth", "10", "sine", "300-3000", "vol", "0.5"], 
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(["sox", "-n", "input/test_voice.wav", "synth", "3", "sine", "440", "vol", "0.5"], 
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                input_path = "input/test_sermon.wav"
                reference_voice_path = "input/test_voice.wav"
                print(f"Created test files: {input_path} and {reference_voice_path}")
                
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fallback naar torchaudio als sox niet beschikbaar is
                print("Sox not available, using torchaudio to generate test files")
                
                # Maak een testaudio bestand met torchaudio (pure toon)
                sample_rate = 16000
                test_audio = torch.sin(2 * np.pi * 440 * torch.arange(sample_rate * 3) / sample_rate).unsqueeze(0)
                sermon_audio = torch.sin(2 * np.pi * 300 * torch.arange(sample_rate * 10) / sample_rate).unsqueeze(0)
                
                torchaudio.save("input/test_voice.wav", test_audio, sample_rate)
                torchaudio.save("input/test_sermon.wav", sermon_audio, sample_rate)
                
                input_path = "input/test_sermon.wav" 
                reference_voice_path = "input/test_voice.wav"
                print(f"Created test files: {input_path} and {reference_voice_path}")
        
        except Exception as e:
            print(f"Error creating test files: {e}")
            print("Continuing with regular mode")
    
    # Check of bestanden bestaan
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        print("Please place your Dutch sermon audio in the input directory")
        return
    
    # Maak output directory als nodig
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Controleer of stem-referentie bestaat
    use_voice_cloning = os.path.exists(reference_voice_path)
    if use_voice_cloning:
        print(f"Using reference voice from: {reference_voice_path}")
        print("Voice reference will be used for translation")
    else:
        print(f"Warning: Voice reference file not found: {reference_voice_path}")
        print("Continuing without voice cloning...")
        reference_voice_path = None
    
    # Bepaal de device (mps, cuda, of cpu)
    if args.device:
        device = args.device
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # Toon systeeminformatie
    print(f"System info:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- Device: {device}")
    if device == "cuda":
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
    print(f"- Input file: {input_path}")
    print(f"- Voice reference: {reference_voice_path}")
    print(f"- Output file: {output_path}")
    print(f"- Source language: {args.src_lang}")
    print(f"- Target language: {args.tgt_lang}")
    print(f"- Chunk size: {args.chunk_size}ms")
    
    # Initialiseer de translator
    translator = SeamlessS2ST(device=device, dtype=torch.float32)
    translator.chunk_size_ms = args.chunk_size
    
    # Voer de vertaling uit
    try:
        translator.translate_long_audio(
            input_path=input_path,
            output_path=output_path,
            reference_audio_path=reference_voice_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        print("\nMVP Translation Process Complete!")
        print("----------------------------")
        print(f"Output file: {output_path}")
        
    except Exception as e:
        print(f"Error during translation: {e}")
        traceback_str = traceback.format_exc()
        print(f"Detailed traceback:\n{traceback_str}")
        print("Translation failed. Check the error messages above for details.")


if __name__ == "__main__":
    main()