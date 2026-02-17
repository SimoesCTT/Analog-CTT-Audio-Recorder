#!/usr/bin/env python3
"""
🎧 CTT 4-TRACK STUDIO — FFT EDITION
Using proper spectral analysis instead of Goertzel
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
import json
from datetime import datetime
from scipy import signal
from scipy.signal import savgol_filter, stft, istft
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

PYDUB_AVAILABLE = False
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
    print("✅ pydub loaded successfully")
except ImportError as e:
    print(f"⚠️  pydub not available: {e}")

FFMPEG_AVAILABLE = False
if PYDUB_AVAILABLE:
    try:
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            FFMPEG_AVAILABLE = True
            print(f"✅ ffmpeg found at: {ffmpeg_path}")
        else:
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                FFMPEG_AVAILABLE = True
                print(f"✅ ffmpeg found at: {result.stdout.strip()}")
    except Exception as e:
        print(f"⚠️  Error checking ffmpeg: {e}")

# ============================================================================
# MICROPHONE DETECTION
# ============================================================================

def detect_microphones():
    """Scan and display available microphones"""
    print("\n🎤 Scanning for microphones...")
    print("="*60)
    
    devices = sd.query_devices()
    mics = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            mics.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'default_sr': int(device['default_samplerate'])
            })
            print(f"\n  [{len(mics)}] {device['name']}")
            print(f"      Channels: {device['max_input_channels']}")
            print(f"      Sample rate: {int(device['default_samplerate'])} Hz")
    
    print("\n" + "="*60)
    return mics

def select_microphone(mics):
    """Let user select which microphone to use"""
    if not mics:
        print("❌ No microphones found!")
        return None
    
    if len(mics) == 1:
        print(f"\n✅ Using only available microphone: {mics[0]['name']}")
        return mics[0]
    
    print("\n🎤 Multiple microphones found:")
    for i, mic in enumerate(mics):
        print(f"   {i+1}. {mic['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect microphone (1-{len(mics)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(mics):
                selected = mics[idx]
                print(f"✅ Using: {selected['name']}")
                return selected
        except:
            pass
        print("❌ Invalid selection")

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

def generate_frequencies(n_zeros=480, mic_range=None):
    """Generate frequency distribution (kept for metadata)"""
    if mic_range:
        low, high = mic_range
    else:
        low, high = 30, 20000
    
    # Just return the range for display
    return np.array([low, high])

# ============================================================================
# AUDIO LOADING
# ============================================================================

def load_audio_file(filepath, target_sr=44100):
    """Load audio from various formats"""
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()
    
    print(f"   File extension: {ext}")
    
    try:
        audio, sr = sf.read(filepath)
        print(f"   ✅ Loaded with soundfile: {sr}Hz")
        return audio, sr
    except Exception as e:
        print(f"   soundfile failed: {e}")
    
    if PYDUB_AVAILABLE:
        try:
            print(f"   🔄 Trying pydub...")
            audio_segment = AudioSegment.from_file(filepath)
            sr = audio_segment.frame_rate
            channels = audio_segment.channels
            
            print(f"   ✅ Loaded with pydub: {sr}Hz, {channels} channels")
            
            if channels == 2:
                samples = np.array(audio_segment.get_array_of_samples())
                audio = samples.reshape((-1, 2)).mean(axis=1)
            else:
                audio = np.array(audio_segment.get_array_of_samples())
            
            sample_width = audio_segment.sample_width
            max_val = float(2**(8 * sample_width - 1))
            audio = audio.astype(np.float32) / max_val
            
            return audio, sr
            
        except Exception as e:
            print(f"   ❌ pydub error: {e}")
    
    try:
        print(f"   🔄 Trying direct ffmpeg...")
        temp_wav = f"/tmp/ctt_temp_{int(time.time())}.wav"
        cmd = ['ffmpeg', '-i', filepath, '-acodec', 'pcm_s16le', '-ar', str(target_sr), '-ac', '1', '-y', temp_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_wav):
            audio, sr = sf.read(temp_wav)
            os.remove(temp_wav)
            print(f"   ✅ Loaded with ffmpeg: {sr}Hz")
            return audio, sr
    except Exception as e:
        print(f"   ffmpeg error: {e}")
    
    raise ValueError(f"Could not load {filepath}")

# ============================================================================
# CTT STUDIO - FFT VERSION
# ============================================================================

class CTTStudio:
    def __init__(self, sample_rate=44100, n_zeros=480, mic=None):
        self.sr = sample_rate
        self.n_zeros = n_zeros
        self.mic = mic
        
        # FFT parameters
        self.fft_size = 2048  # Good balance of time/frequency resolution
        self.hop_size = self.fft_size // 4  # 75% overlap for smooth reconstruction
        self.window = signal.windows.hann(self.fft_size)
        
        print(f"\n🎵 Using FFT analysis: {self.fft_size} point FFT")
        print(f"   Frequency resolution: {self.sr/self.fft_size:.1f}Hz")
        print(f"   Time resolution: {self.fft_size/self.sr*1000:.1f}ms")
        
        self.n_cores = mp.cpu_count()
        print(f"⚡ Using {self.n_cores} CPU cores")
        
        # Tracks - now storing FFT frames instead of Goertzel results
        self.tracks = [None] * 4
        self.track_names = ["Track 1", "Track 2", "Track 3", "Track 4"]
        self.track_metadata = [None] * 4
        self.session_dir = None
        
        # Noise floor (in FFT space)
        self.noise_floor = None
    
    def calibrate_noise(self, duration=2):
        """Calibrate noise floor using FFT"""
        print(f"\n🔇 Calibrating noise floor ({duration}s)...")
        if self.mic:
            print(f"   Using: {self.mic['name']}")
        print("   Please be quiet...")
        time.sleep(1)
        
        if self.mic:
            sd.default.device[0] = self.mic['index']
        
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1)
        sd.wait()
        audio = audio.flatten()
        
        # Compute FFT of silence to get noise floor
        f, t, Zxx = signal.stft(audio, self.sr, window='hann', 
                                 nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
        
        # Average magnitude across time
        self.noise_floor = np.mean(np.abs(Zxx), axis=1)
        
        sd.default.device = None
        
        avg_noise = np.mean(self.noise_floor)
        print(f"✅ Noise floor calibrated (avg: {avg_noise:.6f})")
    
    def create_session(self):
        """Create new session"""
        self.session_dir = f"ctt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fft"
        os.makedirs(self.session_dir, exist_ok=True)
        abs_path = os.path.abspath(self.session_dir)
        print(f"\n📁 Session created: {abs_path}")
        self.calibrate_noise()
    
    def load_session(self, path):
        """Load existing session"""
        if not os.path.exists(path):
            print(f"❌ Not found: {path}")
            return False
        
        self.session_dir = path
        abs_path = os.path.abspath(path)
        print(f"\n📂 Loaded session: {abs_path}")
        
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
                        print(f"✅ Found Track {i+1}: {meta.get('name', 'Unknown')}")
                except:
                    pass
        
        return True
    
    def record_track(self, track_num):
        """Record a new track using FFT"""
        if track_num < 1 or track_num > 4:
            print("❌ Invalid track")
            return
        
        if self.session_dir is None:
            self.create_session()
        
        idx = track_num - 1
        
        if self.mic:
            sd.default.device[0] = self.mic['index']
            mic_name = self.mic['name']
        else:
            mic_name = "default microphone"
        
        name = input(f"Track name [{self.track_names[idx]}]: ").strip()
        if name:
            self.track_names[idx] = name
        
        gain = 0.5
        gain_input = input(f"Input gain (0.1-1.0) [0.5]: ").strip()
        if gain_input:
            try:
                gain = float(gain_input)
                gain = max(0.1, min(1.0, gain))
            except:
                pass
        
        print(f"\n🎤 Recording {self.track_names[idx]} — Ctrl+C to stop")
        print(f"   Using: {mic_name}")
        print(f"   Input gain: {gain:.2f}")
        
        audio = []
        clipping = False
        
        def callback(indata, frames, time, status):
            nonlocal clipping
            data = indata.copy().flatten() * gain
            audio.extend(data)
            duration = len(audio)/self.sr
            level = np.max(np.abs(data))
            
            if level > 0.95:
                clipping = True
            
            bars = int(level * 50)
            meter = '█' * bars + '░' * (50 - bars)
            clip_warn = " ⚠️ CLIPPING!" if clipping else ""
            print(f"   {duration:.1f}s | Level: {level:.3f} [{meter}]{clip_warn}", end='\r')
        
        stream = sd.InputStream(samplerate=self.sr, channels=1, 
                               callback=callback, blocksize=4096)
        with stream:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopped")
        
        sd.default.device = None
        
        if len(audio) == 0:
            print("❌ No audio recorded")
            return
        
        audio = np.array(audio, dtype=np.float32)
        duration = len(audio) / self.sr
        
        if clipping:
            print("⚠️  Warning: Audio was clipping!")
        
        print(f"⏱️  Processing {duration:.1f}s with FFT...")
        start_time = time.time()
        
        # Compute STFT
        f, t, Zxx = signal.stft(audio, self.sr, window='hann', 
                                 nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
        
        # Apply noise gate
        if self.noise_floor is not None:
            # Create mask for frequencies above noise floor
            mask = np.abs(Zxx) > self.noise_floor[:, np.newaxis] * 1.5
            Zxx = Zxx * mask
        
        proc_time = time.time() - start_time
        speed = duration / proc_time
        print(f"⚡ Processed in {proc_time:.2f}s ({speed:.1f}x realtime)")
        
        # Store FFT data
        self.tracks[idx] = (f, t, Zxx)
        self.track_metadata[idx] = {
            'track': track_num,
            'name': self.track_names[idx],
            'duration': duration,
            'source': f'mic: {mic_name}',
            'gain': gain,
            'clipping': clipping,
            'fft_size': self.fft_size,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_track(idx)
        print(f"✅ Track {track_num} saved")
    
    def import_file(self, filepath, track_num=None):
        """Import any audio file and convert to FFT representation"""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        if track_num is None:
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
        
        try:
            audio, original_sr = load_audio_file(filepath, self.sr)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            print(f"   Converted to mono")
        
        if original_sr != self.sr:
            print(f"   Resampling from {original_sr}Hz to {self.sr}Hz")
            new_length = int(len(audio) * self.sr / original_sr)
            audio = signal.resample(audio, new_length)
        
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        
        duration = len(audio) / self.sr
        print(f"   Duration: {duration:.1f}s")
        
        default_name = Path(filepath).stem
        default_name = ''.join(c for c in default_name if c.isalnum() or c in ' -_').strip()
        name = input(f"Track name [{default_name}]: ").strip()
        if not name:
            name = default_name
        self.track_names[idx] = name
        
        print(f"⏱️  Converting to FFT representation...")
        start_time = time.time()
        
        # Compute STFT
        f, t, Zxx = signal.stft(audio, self.sr, window='hann', 
                                 nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
        
        # Optional noise gate
        if self.noise_floor is not None:
            mask = np.abs(Zxx) > self.noise_floor[:, np.newaxis] * 1.2
            Zxx = Zxx * mask
        
        proc_time = time.time() - start_time
        speed = duration / proc_time
        print(f"⚡ Converted in {proc_time:.2f}s ({speed:.1f}x realtime)")
        
        # Store FFT data
        self.tracks[idx] = (f, t, Zxx)
        self.track_metadata[idx] = {
            'track': track_num,
            'name': name,
            'duration': duration,
            'source': str(filepath),
            'original_sr': original_sr,
            'fft_size': self.fft_size,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.session_dir is None:
            self.create_session()
        
        self._save_track(idx)
        print(f"✅ Imported to Track {track_num}")
        
        return True
    
    def _save_track(self, idx):
        """Save FFT data to disk"""
        if self.tracks[idx] is None:
            return
        
        f, t, Zxx = self.tracks[idx]
        base_fname = f"{self.session_dir}/track_{idx+1}"
        meta = self.track_metadata[idx]
        
        # Save FFT data (real and imaginary parts separately for compression)
        np.savez_compressed(base_fname, 
                           f=f,
                           t=t,
                           real=np.real(Zxx).astype(np.float16),
                           imag=np.imag(Zxx).astype(np.float16),
                           metadata=json.dumps(meta))
        
        npz_file = f"{base_fname}.npz"
        ctt_file = f"{base_fname}.ctt"
        if os.path.exists(npz_file):
            os.rename(npz_file, ctt_file)
            size_mb = os.path.getsize(ctt_file) / (1024 * 1024)
            print(f"💾 Saved: {ctt_file} ({size_mb:.1f} MB)")
        
        if self.noise_floor is not None:
            np.save(f"{self.session_dir}/noise_floor.npy", self.noise_floor)
    
    def _load_track(self, track_num):
        """Load FFT data from disk"""
        fname = f"{self.session_dir}/track_{track_num}.ctt"
        if not os.path.exists(fname):
            return None, None, None, None
        
        data = np.load(fname, allow_pickle=True)
        f = data['f']
        t = data['t']
        Zxx = data['real'] + 1j * data['imag']
        meta = json.loads(data['metadata'].item())
        return f, t, Zxx, meta
    
    def play_track(self, track_num):
        """Play a track by reconstructing from FFT"""
        if track_num < 1 or track_num > 4:
            print("❌ Invalid track")
            return
        
        idx = track_num - 1
        
        if self.tracks[idx] is None and self.session_dir:
            print(f"⏱️  Loading track from disk...")
            f, t, Zxx, meta = self._load_track(track_num)
            if f is not None:
                self.tracks[idx] = (f, t, Zxx)
                self.track_names[idx] = meta['name']
                self.track_metadata[idx] = meta
                print(f"   Loaded: {Zxx.shape[1]} time frames, {Zxx.shape[0]} frequency bins")
        
        if self.tracks[idx] is None:
            print("❌ Track empty")
            return
        
        f, t, Zxx = self.tracks[idx]
        print(f"\n🔊 Playing {self.track_names[idx]}...")
        
        start_time = time.time()
        
        # Inverse STFT to reconstruct audio
        _, audio = signal.istft(Zxx, self.sr, window='hann', 
                                 nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
        
        decode_time = time.time() - start_time
        duration = len(audio) / self.sr
        
        print(f"   Decoded in {decode_time:.2f}s ({duration/decode_time:.1f}x realtime)")
        
        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        
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
        """Export FFT track to WAV"""
        if track_num is not None:
            if track_num < 1 or track_num > 4:
                print("❌ Invalid track")
                return
            
            idx = track_num - 1
            
            if self.tracks[idx] is None and self.session_dir:
                f, t, Zxx, meta = self._load_track(track_num)
                if f is not None:
                    self.tracks[idx] = (f, t, Zxx)
                    self.track_names[idx] = meta['name']
            
            if self.tracks[idx] is None:
                print("❌ Track empty")
                return
            
            f, t, Zxx = self.tracks[idx]
            print(f"\n💿 Exporting {self.track_names[idx]}...")
            
            start_time = time.time()
            
            # Inverse STFT
            _, audio = signal.istft(Zxx, self.sr, window='hann', 
                                     nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
            
            decode_time = time.time() - start_time
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95
            
            safe_name = "".join(c for c in self.track_names[idx] if c.isalnum() or c in ' -_').strip()
            fname = f"{self.session_dir}/{safe_name}_fft.wav"
            abs_path = os.path.abspath(fname)
            
            sf.write(fname, audio, self.sr, subtype='PCM_24')
            
            size_mb = os.path.getsize(fname) / (1024 * 1024)
            duration = len(audio) / self.sr
            print(f"✅ Exported: {abs_path}")
            print(f"   Size: {size_mb:.1f} MB | Duration: {duration:.1f}s | Time: {decode_time:.2f}s")
            
        else:
            print("\n💿 Exporting all tracks...")
            exported = 0
            total_start = time.time()
            
            for i in range(4):
                if self.tracks[i] is not None:
                    f, t, Zxx = self.tracks[i]
                    start_time = time.time()
                    
                    _, audio = signal.istft(Zxx, self.sr, window='hann', 
                                             nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
                    
                    decode_time = time.time() - start_time
                    
                    peak = np.max(np.abs(audio))
                    if peak > 0:
                        audio = audio / peak * 0.95
                    
                    safe_name = "".join(c for c in self.track_names[i] if c.isalnum() or c in ' -_').strip()
                    fname = f"{self.session_dir}/{safe_name}_fft.wav"
                    abs_path = os.path.abspath(fname)
                    
                    sf.write(fname, audio, self.sr, subtype='PCM_24')
                    size_mb = os.path.getsize(fname) / (1024 * 1024)
                    duration = len(audio) / self.sr
                    
                    print(f"\n   ✅ Track {i+1}: {os.path.basename(abs_path)}")
                    print(f"      Size: {size_mb:.1f} MB | Duration: {duration:.1f}s | Time: {decode_time:.2f}s")
                    exported += 1
            
            total_time = time.time() - total_start
            if exported > 0:
                print(f"\n📁 Export directory: {os.path.abspath(self.session_dir)}")
                print(f"✅ Exported {exported} tracks in {total_time:.2f}s")
    
    def list_tracks(self):
        """List all tracks"""
        print("\n" + "="*70)
        print("📋 TRACK LIST (FFT)")
        print("="*70)
        
        if self.session_dir:
            print(f"📁 Session: {os.path.abspath(self.session_dir)}")
        
        for i in range(4):
            print(f"\nTrack {i+1}: {self.track_names[i]}")
            print("-" * 40)
            
            if self.tracks[i] is not None:
                f, t, Zxx = self.tracks[i]
                print(f"  Status: ✅ Loaded")
                print(f"  Time frames: {Zxx.shape[1]}")
                print(f"  Frequency bins: {Zxx.shape[0]}")
                print(f"  Resolution: {self.sr/self.fft_size:.1f}Hz")
                
                if self.track_metadata[i]:
                    meta = self.track_metadata[i]
                    print(f"  Duration: {meta.get('duration', 0):.1f}s")
                    print(f"  Source: {meta.get('source', 'unknown')}")
            
            elif self.session_dir and os.path.exists(f"{self.session_dir}/track_{i+1}.ctt"):
                print(f"  Status: 💾 On disk")
            
            else:
                print(f"  Status: ❌ Empty")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CTT 4-Track Studio - FFT Edition')
    parser.add_argument('files', nargs='*', help='Audio files to import')
    parser.add_argument('-t', '--track', type=int, choices=range(1,5), help='Track number (1-4)')
    parser.add_argument('-z', '--zeros', type=int, default=480, help='Number of Riemann zeros (metadata only)')
    parser.add_argument('-s', '--session', help='Session directory')
    parser.add_argument('--play', type=int, choices=range(1,5), help='Play a track')
    parser.add_argument('--export', type=int, nargs='?', const=0, help='Export track(s) to WAV')
    parser.add_argument('--no-mic-opt', action='store_true', help='Disable microphone optimization')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎧 CTT 4-TRACK STUDIO — FFT EDITION")
    print("="*80)
    print("⚡ Using Short-Time Fourier Transform instead of Goertzel")
    print("🎚️  Cleaner reconstruction, fewer artifacts, faster processing")
    
    if PYDUB_AVAILABLE:
        print("📁 Import: WAV, AIFF, FLAC, OGG, MP3, M4A, MP4, WMA, and more!")
    else:
        print("📁 Import: WAV, AIFF, FLAC, OGG, MP3")
    
    print("="*80)
    
    # Detect microphone
    selected_mic = None
    if not args.no_mic_opt:
        mics = detect_microphones()
        selected_mic = select_microphone(mics)
    
    # Initialize studio
    studio = CTTStudio(n_zeros=args.zeros, mic=selected_mic)
    
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
        if studio.mic:
            print(f"🎤 Mic: {studio.mic['name']}")
        print(f"📊 FFT Size: {studio.fft_size} points | Resolution: {studio.sr/studio.fft_size:.1f}Hz")
        if studio.session_dir:
            print(f"📁 Session: {os.path.basename(studio.session_dir)}")
        print("-"*60)
        print("1-4 : Record track")
        print("i   : Import audio file")
        print("p   : Play track")
        print("m   : Play all tracks")
        print("e   : Export to WAV")
        print("l   : List tracks")
        print("c   : Calibrate noise")
        print("n   : New session")
        print("o   : Load session")
        print("q   : Quit")
        print("-"*60)
        
        cmd = input("> ").strip().lower()
        
        if cmd in '1234':
            studio.record_track(int(cmd))
        
        elif cmd == 'i':
            path = input("File path: ").strip()
            if path:
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
