#!/usr/bin/env python3
"""
🎧 CTT 4-TRACK STUDIO — UNIVERSAL EDITION
Convert any audio to Riemann zero representation
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
import json
from datetime import datetime
from scipy import signal
from scipy.signal import savgol_filter
import multiprocessing as mp
import numba
from numba import jit, prange
import argparse
from pathlib import Path
import subprocess

# ============================================================================
# CHECK DEPENDENCIES AT STARTUP
# ============================================================================

print("\n🔍 Checking dependencies...")

# Check pydub
PYDUB_AVAILABLE = False
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    print("✅ pydub loaded successfully")
except ImportError as e:
    print(f"⚠️  pydub not available: {e}")
    print("   Install with: pip install pydub")

# Check ffmpeg if pydub is available
FFMPEG_AVAILABLE = False
if PYDUB_AVAILABLE:
    try:
        # Try to find ffmpeg in system PATH
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            FFMPEG_AVAILABLE = True
            print(f"✅ ffmpeg found at: {ffmpeg_path}")
        else:
            # Also try which command directly
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                FFMPEG_AVAILABLE = True
                print(f"✅ ffmpeg found at: {result.stdout.strip()}")
            else:
                print("⚠️  ffmpeg not found in PATH")
                print("   M4A/MP3 files may not work")
                print("   Install ffmpeg:")
                print("   - Fedora: sudo dnf install ffmpeg")
                print("   - Ubuntu: sudo apt install ffmpeg")
    except Exception as e:
        print(f"⚠️  Error checking ffmpeg: {e}")

# ============================================================================
# CTT CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
ALPHA_RH = np.log(PHI) / (2 * np.pi)
TAU_W = 11e-9

# Original 24 Riemann zeros
RIEMANN_ZEROS_24 = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 48.005151, 49.773832, 52.970321,
    56.446248, 59.347044, 60.831779, 65.112544, 67.079811,
    69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 86.970000, 87.425275
])

def generate_frequencies(n_zeros=480):
    """Generate frequency distribution from Riemann zeros"""
    zeros = np.sort(RIEMANN_ZEROS_24)
    
    # Harmonic expansion
    harmonics = []
    for z in zeros:
        for h in range(1, 9):  # Up to 8th harmonic
            harmonics.append(z * h)
    
    harmonics = np.array(harmonics)
    harmonics = np.sort(harmonics)
    harmonics = np.unique(harmonics)
    
    # Get exactly n_zeros frequencies
    if len(harmonics) > n_zeros:
        indices = np.linspace(0, len(harmonics)-1, n_zeros, dtype=int)
        zeros_n = harmonics[indices]
    else:
        x_orig = np.arange(len(harmonics))
        x_target = np.linspace(0, len(harmonics)-1, n_zeros)
        zeros_n = np.interp(x_target, x_orig, harmonics)
    
    # Map to audio frequencies
    freq_min = 55   # A1
    freq_max = 3520 # A7
    freqs = freq_min * (freq_max/freq_min) ** (zeros_n / np.max(zeros_n))
    
    return freqs

# Default to 480 zeros for good balance
AUDIO_FREQS = generate_frequencies(480)

# ============================================================================
# AUDIO LOADING WITH MULTIPLE FORMAT SUPPORT
# ============================================================================

def load_audio_file(filepath, target_sr=44100):
    """
    Load audio from various formats using available backends
    Returns (audio_array, original_sr)
    """
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    
    print(f"   File extension: {ext}")
    
    # Try soundfile first (WAV, AIFF, FLAC, OGG, MP3)
    try:
        audio, sr = sf.read(filepath)
        print(f"   ✅ Loaded with soundfile: {sr}Hz")
        return audio, sr
    except Exception as e:
        print(f"   soundfile failed: {e}")
    
    # Try pydub if available
    if PYDUB_AVAILABLE:
        try:
            print(f"   🔄 Trying pydub...")
            
            # Load with pydub
            audio_segment = AudioSegment.from_file(filepath)
            sr = audio_segment.frame_rate
            channels = audio_segment.channels
            
            print(f"   ✅ Loaded with pydub: {sr}Hz, {channels} channels")
            
            # Convert to numpy array
            if channels == 2:
                # Convert stereo to mono by averaging
                samples = np.array(audio_segment.get_array_of_samples())
                audio = samples.reshape((-1, 2)).mean(axis=1)
            else:
                audio = np.array(audio_segment.get_array_of_samples())
            
            # Convert to float32 in range [-1, 1]
            sample_width = audio_segment.sample_width
            max_val = float(2**(8 * sample_width - 1))
            audio = audio.astype(np.float32) / max_val
            
            return audio, sr
            
        except Exception as e:
            print(f"   ❌ pydub error: {e}")
            if "ffmpeg" in str(e).lower():
                print("   ⚠️  This appears to be an ffmpeg issue")
                print("   💡 On Fedora: sudo dnf install ffmpeg")
                print("   💡 On Ubuntu: sudo apt install ffmpeg")
    
    # Try using ffmpeg directly as last resort
    try:
        print(f"   🔄 Trying direct ffmpeg...")
        # Create a temporary wav file
        temp_wav = f"/tmp/ctt_temp_{int(time.time())}.wav"
        cmd = ['ffmpeg', '-i', filepath, '-acodec', 'pcm_s16le', '-ar', str(target_sr), '-ac', '1', '-y', temp_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_wav):
            audio, sr = sf.read(temp_wav)
            os.remove(temp_wav)
            print(f"   ✅ Loaded with ffmpeg: {sr}Hz")
            return audio, sr
        else:
            print(f"   ffmpeg failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"   ffmpeg error: {e}")
    
    # If we get here, no backend worked
    supported = ["WAV", "AIFF", "FLAC", "OGG", "MP3"]
    if PYDUB_AVAILABLE:
        supported.extend(["M4A", "MP4", "WMA"])
    
    error_msg = f"Could not load {filepath}. Supported formats: {', '.join(supported)}"
    if not FFMPEG_AVAILABLE and ext in ['.m4a', '.mp4']:
        error_msg += "\n   💡 This file type requires ffmpeg. Install it with: sudo dnf install ffmpeg"
    
    raise ValueError(error_msg)

# ============================================================================
# OPTIMIZED PROCESSING
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def goertzel_batch(samples, freqs, sample_rate, window_size):
    """Batch Goertzel algorithm"""
    n_freqs = len(freqs)
    phases = np.zeros(n_freqs)
    magnitudes = np.zeros(n_freqs)
    
    k_float = window_size * freqs / sample_rate
    omega = 2 * np.pi * k_float / window_size
    coeff = 2 * np.cos(omega)
    
    for f in prange(n_freqs):
        s_prev = 0.0
        s_prev2 = 0.0
        c = coeff[f]
        
        for sample in samples:
            s = sample + c * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        
        real = s_prev - s_prev2 * np.cos(omega[f])
        imag = s_prev2 * np.sin(omega[f])
        magnitudes[f] = np.sqrt(real*real + imag*imag) / window_size
        phases[f] = np.arctan2(imag, real)
    
    return phases, magnitudes

@jit(nopython=True, parallel=True, cache=True)
def synthesize_batch(phases, magnitudes, freqs, sample_rate, window_size, hop_size, n_windows):
    """Batch synthesis"""
    total_samples = n_windows * hop_size + window_size
    audio = np.zeros(total_samples)
    overlap = np.zeros(total_samples)
    window = np.hanning(window_size)
    
    t = np.arange(window_size) / sample_rate
    
    for w in prange(n_windows):
        start = w * hop_size
        
        window_out = np.zeros(window_size)
        for f in range(len(freqs)):
            if magnitudes[w, f] > 1e-8:
                window_out += magnitudes[w, f] * np.sin(
                    2 * np.pi * freqs[f] * t + phases[w, f]
                )
        
        for i in range(window_size):
            idx = start + i
            if idx < total_samples:
                audio[idx] += window_out[i] * window[i]
                overlap[idx] += window[i]
    
    return audio, overlap


# ============================================================================
# CTT STUDIO
# ============================================================================

class CTTStudio:
    def __init__(self, sample_rate=44100, n_zeros=480):
        self.sr = sample_rate
        self.n_zeros = n_zeros
        self.freqs = generate_frequencies(n_zeros)
        self.n_freqs = len(self.freqs)
        
        print(f"\n🎵 Riemann Zeros: {self.n_freqs}")
        print(f"   Frequency range: {self.freqs[0]:.1f}Hz - {self.freqs[-1]:.1f}Hz")
        
        # Processing parameters
        self.window_ms = 46
        self.window_size = int(sample_rate * self.window_ms / 1000)
        self.window_size = 2**int(np.log2(self.window_size))
        self.hop_size = self.window_size // 4
        self.window = np.hanning(self.window_size).astype(np.float32)
        
        self.n_cores = mp.cpu_count()
        print(f"⚡ Using {self.n_cores} CPU cores")
        
        # Tracks
        self.tracks = [None] * 4
        self.track_names = ["Track 1", "Track 2", "Track 3", "Track 4"]
        self.track_metadata = [None] * 4
        self.session_dir = None
        
        # Noise floor (will be calibrated)
        self.noise_floor = None
        
        # Survival probabilities
        self.survival_probs = np.exp(-ALPHA_RH * self.freqs * TAU_W)
    
    def calibrate_noise(self, duration=2):
        """Calibrate noise floor"""
        print(f"\n🔇 Calibrating noise floor ({duration}s)...")
        print("   Please be quiet...")
        time.sleep(1)
        
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1)
        sd.wait()
        audio = audio.flatten()
        
        # Analyze noise
        n_windows = 20
        noise_samples = []
        
        for w in range(n_windows):
            start = w * self.hop_size
            if start + self.window_size > len(audio):
                break
            window = audio[start:start + self.window_size] * self.window
            _, mags = goertzel_batch(window, self.freqs, self.sr, self.window_size)
            noise_samples.append(mags)
        
        noise_samples = np.array(noise_samples)
        self.noise_floor = np.mean(noise_samples, axis=0) + 2 * np.std(noise_samples, axis=0)
        
        print(f"✅ Noise floor calibrated")
    
    def create_session(self):
        """Create new session"""
        self.session_dir = f"ctt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"\n📁 Session: {self.session_dir}/")
        self.calibrate_noise()
    
    def load_session(self, path):
        """Load existing session"""
        if not os.path.exists(path):
            print(f"❌ Not found: {path}")
            return False
        
        self.session_dir = path
        
        # Load noise floor
        noise_file = f"{path}/noise_floor.npy"
        if os.path.exists(noise_file):
            self.noise_floor = np.load(noise_file)
            print(f"✅ Loaded noise floor")
        
        # Load track metadata
        for i in range(4):
            fname = f"{path}/track_{i+1}.ctt"
            if os.path.exists(fname):
                try:
                    data = np.load(fname, allow_pickle=True)
                    if 'metadata' in data:
                        meta = json.loads(data['metadata'].item())
                        self.track_names[i] = meta.get('name', self.track_names[i])
                        self.track_metadata[i] = meta
                except:
                    pass
        
        print(f"📂 Loaded: {path}")
        return True
    
    def record_track(self, track_num):
        """Record a new track"""
        if track_num < 1 or track_num > 4:
            print("❌ Invalid track")
            return
        
        if self.session_dir is None:
            self.create_session()
        
        idx = track_num - 1
        
        # Get track name
        name = input(f"Track name [{self.track_names[idx]}]: ").strip()
        if name:
            self.track_names[idx] = name
        
        print(f"\n🎤 Recording {self.track_names[idx]} — Ctrl+C to stop")
        
        # Record audio
        audio = []
        def callback(indata, frames, time, status):
            audio.extend(indata.copy().flatten())
            duration = len(audio)/self.sr
            print(f"   {duration:.1f}s", end='\r')
        
        stream = sd.InputStream(samplerate=self.sr, channels=1, 
                               callback=callback, blocksize=4096)
        with stream:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopped")
        
        if len(audio) == 0:
            print("❌ No audio recorded")
            return
        
        audio = np.array(audio, dtype=np.float32)
        duration = len(audio) / self.sr
        
        print(f"⏱️  Processing {duration:.1f}s...")
        start_time = time.time()
        
        phases, amps = self._audio_to_ctt(audio)
        
        proc_time = time.time() - start_time
        print(f"⚡ Processed in {proc_time:.2f}s")
        
        self.tracks[idx] = (phases, amps)
        self.track_metadata[idx] = {
            'track': track_num,
            'name': self.track_names[idx],
            'duration': duration,
            'source': 'microphone',
            'n_zeros': self.n_freqs,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_track(idx)
        print(f"✅ Track {track_num} saved")
    
    def import_file(self, filepath, track_num=None):
        """Import any audio file and convert to CTT"""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        # Determine track number
        if track_num is None:
            # Find first empty track
            for i in range(4):
                if self.tracks[i] is None:
                    track_num = i + 1
                    break
            if track_num is None:
                print("❌ All tracks full")
                return False
        
        if track_num < 1 or track_num > 4:
            print("❌ Invalid track number")
            return False
        
        idx = track_num - 1
        
        print(f"\n📂 Importing: {filepath}")
        
        # Load audio file with multi-format support
        try:
            audio, original_sr = load_audio_file(filepath, self.sr)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            print(f"   Converted to mono")
        
        # Resample if needed
        if original_sr != self.sr:
            print(f"   Resampling from {original_sr}Hz to {self.sr}Hz")
            # Calculate new length
            new_length = int(len(audio) * self.sr / original_sr)
            audio = signal.resample(audio, new_length)
        
        # Normalize
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        
        duration = len(audio) / self.sr
        print(f"   Duration: {duration:.1f}s")
        
        # Get track name from filename
        default_name = Path(filepath).stem
        # Clean up name (remove any brackets, etc.)
        default_name = ''.join(c for c in default_name if c.isalnum() or c in ' -_').strip()
        name = input(f"Track name [{default_name}]: ").strip()
        if not name:
            name = default_name
        self.track_names[idx] = name
        
        print(f"⏱️  Converting to CTT format...")
        start_time = time.time()
        
        phases, amps = self._audio_to_ctt(audio)
        
        proc_time = time.time() - start_time
        print(f"⚡ Converted in {proc_time:.2f}s")
        
        # Count active frequencies
        active_freqs = np.sum(amps > 0, axis=1)
        print(f"   Avg active frequencies per window: {np.mean(active_freqs):.1f}")
        
        self.tracks[idx] = (phases, amps)
        self.track_metadata[idx] = {
            'track': track_num,
            'name': name,
            'duration': duration,
            'source': str(filepath),
            'original_sr': original_sr,
            'n_zeros': self.n_freqs,
            'avg_active': float(np.mean(active_freqs)),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.session_dir is None:
            self.create_session()
        
        self._save_track(idx)
        print(f"✅ Imported to Track {track_num}")
        
        return True
    
    def _audio_to_ctt(self, audio):
        """Convert audio to CTT representation"""
        # Pad for analysis
        if len(audio) % self.hop_size != 0:
            pad = self.hop_size - (len(audio) % self.hop_size)
            audio = np.pad(audio, (0, pad))
        
        n_windows = (len(audio) - self.window_size) // self.hop_size + 1
        print(f"   Analyzing {n_windows} windows...")
        
        phases = np.zeros((n_windows, self.n_freqs), dtype=np.float32)
        amps = np.zeros((n_windows, self.n_freqs), dtype=np.float32)
        
        for w in range(n_windows):
            start = w * self.hop_size
            window = audio[start:start + self.window_size] * self.window
            
            phase_vec, mag_vec = goertzel_batch(window, self.freqs, self.sr, self.window_size)
            
            # Apply noise gate if calibrated
            if self.noise_floor is not None:
                mask = mag_vec > self.noise_floor
            else:
                mask = mag_vec > 1e-5
            
            mask = mask & (self.survival_probs > 0.1)
            
            phases[w, mask] = phase_vec[mask]
            amps[w, mask] = mag_vec[mask]
            
            if w % 50 == 0:
                print(f"   Progress: {w}/{n_windows}", end='\r')
        
        print(f"   Progress: {n_windows}/{n_windows} done")
        
        return phases, amps
    
    def _ctt_to_audio(self, phases, amps):
        """Convert CTT back to audio"""
        n_windows = phases.shape[0]
        
        # Gentle fade
        envelope = np.ones(n_windows)
        envelope[:5] = np.linspace(0, 1, 5)
        envelope[-5:] = np.linspace(1, 0, 5)
        amps = amps * envelope[:, np.newaxis]
        
        # Synthesize
        audio, overlap = synthesize_batch(
            phases, amps, self.freqs, self.sr,
            self.window_size, self.hop_size, n_windows
        )
        
        # Normalize overlap
        overlap[overlap < 0.01] = 1
        audio = audio / overlap
        
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.95 / peak)
        
        return audio
    
    def _save_track(self, idx):
        """Save track to disk"""
        if self.tracks[idx] is None:
            return
        
        phases, amps = self.tracks[idx]
        # Use base filename without extension - npz will be added automatically
        base_fname = f"{self.session_dir}/track_{idx+1}"
        meta = self.track_metadata[idx]
        
        # Save as npz first (compressed numpy format)
        np.savez_compressed(base_fname, 
                           phases=phases.astype(np.float16),
                           amplitudes=amps.astype(np.float16), 
                           metadata=json.dumps(meta))
        
        # Rename to .ctt extension
        npz_file = f"{base_fname}.npz"
        ctt_file = f"{base_fname}.ctt"
        if os.path.exists(npz_file):
            os.rename(npz_file, ctt_file)
            print(f"💾 Saved: {ctt_file}")
        
        # Save noise floor
        if self.noise_floor is not None:
            np.save(f"{self.session_dir}/noise_floor.npy", self.noise_floor)
    
    def _load_track(self, track_num):
        """Load track from disk"""
        fname = f"{self.session_dir}/track_{track_num}.ctt"
        if not os.path.exists(fname):
            return None, None, None
        
        # Load the .ctt file (which is actually a renamed .npz)
        data = np.load(fname, allow_pickle=True)
        phases = data['phases'].astype(np.float32)
        amps = data['amplitudes'].astype(np.float32)
        meta = json.loads(data['metadata'].item())
        return phases, amps, meta
    
    def play_track(self, track_num):
        """Play a track"""
        if track_num < 1 or track_num > 4:
            print("❌ Invalid track")
            return
        
        idx = track_num - 1
        
        # Load if needed
        if self.tracks[idx] is None and self.session_dir:
            phases, amps, meta = self._load_track(track_num)
            if phases is not None:
                self.tracks[idx] = (phases, amps)
                self.track_names[idx] = meta['name']
                self.track_metadata[idx] = meta
        
        if self.tracks[idx] is None:
            print("❌ Track empty")
            return
        
        phases, amps = self.tracks[idx]
        print(f"\n🔊 Playing {self.track_names[idx]}...")
        
        audio = self._ctt_to_audio(phases, amps)
        
        # Show info
        if self.track_metadata[idx] and 'avg_active' in self.track_metadata[idx]:
            print(f"   Active frequencies per window: {self.track_metadata[idx]['avg_active']:.1f}")
        
        sd.play(audio, self.sr)
        sd.wait()
    
    def play_all(self):
        """Play all tracks in sequence"""
        print("\n🔊 Playing all tracks...")
        for i in range(1, 5):
            if self.tracks[i-1] is not None:
                print(f"\n--- Track {i}: {self.track_names[i-1]} ---")
                self.play_track(i)
                time.sleep(0.5)
    
    def export_to_wav(self, track_num=None):
        """Export CTT to WAV file"""
        if track_num is not None:
            # Export single track
            if track_num < 1 or track_num > 4:
                print("❌ Invalid track")
                return
            
            idx = track_num - 1
            
            # Load if needed
            if self.tracks[idx] is None and self.session_dir:
                phases, amps, meta = self._load_track(track_num)
                if phases is not None:
                    self.tracks[idx] = (phases, amps)
                    self.track_names[idx] = meta['name']
            
            if self.tracks[idx] is None:
                print("❌ Track empty")
                return
            
            phases, amps = self.tracks[idx]
            print(f"\n💿 Exporting {self.track_names[idx]}...")
            
            audio = self._ctt_to_audio(phases, amps)
            
            # Create filename
            safe_name = "".join(c for c in self.track_names[idx] if c.isalnum() or c in ' -_').strip()
            fname = f"{self.session_dir}/{safe_name}_ctt.wav"
            
            sf.write(fname, audio, self.sr, subtype='PCM_24')
            print(f"✅ Exported: {fname}")
        
        else:
            # Export all tracks
            print("\n💿 Exporting all tracks...")
            exported = 0
            for i in range(4):
                if self.tracks[i] is not None:
                    phases, amps = self.tracks[i]
                    audio = self._ctt_to_audio(phases, amps)
                    safe_name = "".join(c for c in self.track_names[i] if c.isalnum() or c in ' -_').strip()
                    fname = f"{self.session_dir}/{safe_name}_ctt.wav"
                    sf.write(fname, audio, self.sr, subtype='PCM_24')
                    print(f"   ✅ Track {i+1}: {fname}")
                    exported += 1
            if exported == 0:
                print("   No tracks to export")
    
    def list_tracks(self):
        """List all tracks"""
        print("\n" + "="*70)
        print("📋 TRACK LIST")
        print("="*70)
        
        for i in range(4):
            print(f"\nTrack {i+1}: {self.track_names[i]}")
            print("-" * 40)
            
            if self.tracks[i] is not None:
                phases, amps = self.tracks[i]
                active = np.mean(np.sum(amps > 0, axis=1))
                print(f"  Status: ✅ Loaded in memory")
                print(f"  Windows: {phases.shape[0]}")
                print(f"  Active frequencies: {active:.1f} per window")
                
                if self.track_metadata[i]:
                    meta = self.track_metadata[i]
                    print(f"  Duration: {meta.get('duration', 0):.1f}s")
                    print(f"  Source: {meta.get('source', 'unknown')}")
                    if 'avg_active' in meta:
                        print(f"  Avg active: {meta['avg_active']:.1f}")
            
            elif self.session_dir and os.path.exists(f"{self.session_dir}/track_{i+1}.ctt"):
                print(f"  Status: 💾 On disk")
                if self.track_metadata[i]:
                    meta = self.track_metadata[i]
                    print(f"  Duration: {meta.get('duration', 0):.1f}s")
                    print(f"  Source: {meta.get('source', 'unknown')}")
            
            else:
                print(f"  Status: ❌ Empty")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CTT 4-Track Studio - Convert any audio to Riemann zeros')
    parser.add_argument('files', nargs='*', help='Audio files to import')
    parser.add_argument('-t', '--track', type=int, choices=range(1,5), help='Track number (1-4)')
    parser.add_argument('-z', '--zeros', type=int, default=480, help='Number of Riemann zeros (default: 480)')
    parser.add_argument('-s', '--session', help='Session directory')
    parser.add_argument('--play', type=int, choices=range(1,5), help='Play a track')
    parser.add_argument('--export', type=int, nargs='?', const=0, help='Export track(s) to WAV')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎧 CTT 4-TRACK STUDIO — UNIVERSAL EDITION")
    print("="*80)
    print("⚡ Convert any audio to Riemann zero representation")
    
    if PYDUB_AVAILABLE:
        print("📁 Import: WAV, AIFF, FLAC, OGG, MP3, M4A, MP4, WMA, and more!")
        if not FFMPEG_AVAILABLE:
            print("   ⚠️  ffmpeg not found - some formats may not work")
            print("   💡 Install ffmpeg for full support:")
            print("      Fedora: sudo dnf install ffmpeg")
            print("      Ubuntu: sudo apt install ffmpeg")
    else:
        print("📁 Import: WAV, AIFF, FLAC, OGG, MP3 (install pydub for M4A/MP4)")
        print("   💡 pip install pydub")
    
    print("="*80)
    
    # Initialize studio
    studio = CTTStudio(n_zeros=args.zeros)
    
    # Load session if specified
    if args.session:
        studio.load_session(args.session)
    else:
        studio.create_session()
    
    # Import files
    for filepath in args.files:
        studio.import_file(filepath, args.track)
    
    # Play track if requested
    if args.play:
        studio.play_track(args.play)
    
    # Export if requested
    if args.export is not None:
        if args.export == 0:
            studio.export_to_wav()
        else:
            studio.export_to_wav(args.export)
    
    # If no files specified, go interactive
    if not args.files and args.play is None and args.export is None:
        interactive_mode(studio)


def interactive_mode(studio):
    """Interactive menu"""
    while True:
        print("\n" + "-"*60)
        print("🎤 RECORD:")
        print("  1-4    : Record track")
        print("\n📂 IMPORT:")
        print("  i      : Import audio file (WAV, MP3, M4A, FLAC, etc.)")
        print("\n🔊 PLAYBACK:")
        print("  p      : Play track")
        print("  m      : Play all tracks")
        print("\n💿 EXPORT:")
        print("  e      : Export to WAV")
        print("\n🛠️  TOOLS:")
        print("  l      : List tracks")
        print("  c      : Calibrate noise")
        print("  n      : New session")
        print("  o      : Load session")
        print("  q      : Quit")
        print("-"*60)
        
        cmd = input("> ").strip().lower()
        
        if cmd in '1234':
            studio.record_track(int(cmd))
        
        elif cmd == 'i':
            path = input("File path: ").strip()
            if path:
                # Remove quotes if present
                path = path.strip('"\'')
                t = input("Track number (1-4) [auto]: ").strip()
                if t in '1234':
                    studio.import_file(path, int(t))
                else:
                    studio.import_file(path)
        
        elif cmd == 'p':
            try:
                t = int(input("Track (1-4): "))
                studio.play_track(t)
            except:
                print("❌ Invalid input")
        
        elif cmd == 'm':
            studio.play_all()
        
        elif cmd == 'e':
            print("Export options:")
            print("  1-4 : Export single track")
            print("  a   : Export all tracks")
            exp = input("> ").strip().lower()
            if exp in '1234':
                studio.export_to_wav(int(exp))
            elif exp == 'a':
                studio.export_to_wav()
        
        elif cmd == 'l':
            studio.list_tracks()
        
        elif cmd == 'c':
            studio.calibrate_noise()
        
        elif cmd == 'n':
            studio.create_session()
        
        elif cmd == 'o':
            path = input("Session path: ").strip()
            if path:
                studio.load_session(path)
        
        elif cmd == 'q':
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
