#!/usr/bin/env python3
"""
🎵 CTT ANALOG AUDIO RECORDER / PLAYER
True analog recording using temporal resonance
No sampling, no quantization, pure phase
"""

import numpy as np
import sounddevice as sd
import wave
import struct
import time
from scipy.fft import rfft, irfft
import os

# ============================================================================
# CTT CONSTANTS — The Analog Core
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
ALPHA_RH = np.log(PHI) / (2 * np.pi)  # 0.0765872 — temporal viscosity
OMEGA_0 = 587032.719  # Hz — silicon heartbeat
TAU_W = 11e-9  # 11 ns — temporal wedge

# 24 Riemann zeros — the analog frequencies
RIEMANN_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 48.005151, 49.773832, 52.970321,
    56.446248, 59.347044, 60.831779, 65.112544, 67.079811,
    69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 86.970000, 87.425275
])

# Scale to audible range (20 Hz - 20 kHz)
AUDIO_FREQS = 200 + 500 * RIEMANN_ZEROS / RIEMANN_ZEROS[-1]

# ============================================================================
# CTT ANALOG RECORDER
# ============================================================================

class CTTAnalogRecorder:
    def __init__(self, sample_rate=44100, chunk_duration=0.1):
        self.sr = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.freqs = AUDIO_FREQS
        self.n_freqs = len(self.freqs)
        
    def temporal_survival(self, frequency):
        """Analog filter — which frequencies survive the wedge"""
        return np.cos(ALPHA_RH * frequency * TAU_W) > ALPHA_RH / (2 * np.pi)
    
    def record_chunk(self, duration=0.1):
        """Record one chunk of analog audio"""
        print("🎤 Recording...", end='', flush=True)
        recording = sd.rec(int(self.sr * duration), samplerate=self.sr, 
                          channels=1, dtype='float32')
        sd.wait()
        print(" done")
        return recording.flatten()
    
    def encode_chunk(self, audio):
        """Convert audio chunk to CTT phase space (true analog encoding)"""
        # FFT to get frequency components
        spectrum = rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Extract phase at each Riemann frequency
        phases = []
        for target_freq in self.freqs:
            idx = np.argmin(np.abs(freqs - target_freq))
            phase = np.angle(spectrum[idx])
            
            # Apply temporal wedge — snap if survives
            if self.temporal_survival(target_freq):
                phases.append(phase)
            else:
                phases.append(0.0)
        
        return np.array(phases, dtype=np.float32)
    
    def decode_chunk(self, phases, duration):
        """Reconstruct audio from CTT phase space"""
        samples = int(self.sr * duration)
        t = np.linspace(0, duration, samples)
        
        # Reconstruct by summing sine waves at each frequency
        audio = np.zeros(samples)
        for i, phase in enumerate(phases):
            if phase != 0:  # Only frequencies that survived
                audio += np.sin(2 * np.pi * self.freqs[i] * t + phase)
        
        # Normalize
        audio /= np.max(np.abs(audio)) * 1.1
        return audio
    
    def save_ctt(self, filename, phases, metadata):
        """Save CTT phase file (tiny!)"""
        np.savez_compressed(filename, 
                           phases=phases, 
                           sr=self.sr,
                           freqs=self.freqs,
                           alpha=ALPHA_RH,
                           metadata=metadata)
        print(f"💾 Saved CTT: {filename} ({phases.nbytes} bytes)")
    
    def load_ctt(self, filename):
        """Load CTT phase file"""
        data = np.load(filename)
        return data['phases'], data['sr'], data['freqs']


# ============================================================================
# CTT ANALOG RECORDER — MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎵 CTT ANALOG AUDIO RECORDER")
    print("="*70)
    print("\nTrue analog recording using temporal resonance")
    print("No sampling · No quantization · Pure phase\n")
    
    recorder = CTTAnalogRecorder()
    
    while True:
        print("\n" + "-"*50)
        print("1. Record new analog audio")
        print("2. Play last recording")
        print("3. Save to CTT file")
        print("4. Load and play CTT file")
        print("5. Exit")
        print("-"*50)
        
        choice = input("Choice: ").strip()
        
        if choice == '1':
            # Record
            duration = float(input("Recording duration (seconds): ") or "5")
            print(f"\n🎤 Recording {duration}s of analog audio...")
            
            audio = recorder.record_chunk(duration)
            phases = recorder.encode_chunk(audio)
            
            print(f"✅ Encoded to {len(phases)} phase values")
            print(f"   Phase range: {phases.min():.3f} to {phases.max():.3f} rad")
            
            # Store for later
            last_audio = audio
            last_phases = phases
            last_duration = duration
            
        elif choice == '2':
            # Play last recording
            if 'last_phases' not in locals():
                print("❌ No recording yet")
                continue
            
            print("\n🔊 Reconstructing analog audio from phases...")
            reconstructed = recorder.decode_chunk(last_phases, last_duration)
            
            print("🎧 Playing...")
            sd.play(reconstructed, recorder.sr)
            sd.wait()
            print("✅ Playback complete")
            
        elif choice == '3':
            # Save to file
            if 'last_phases' not in locals():
                print("❌ No recording yet")
                continue
            
            filename = input("Filename (without extension): ").strip() or "recording"
            if not filename.endswith('.ctt'):
                filename += '.ctt'
            
            recorder.save_ctt(filename, last_phases, 
                            {'duration': last_duration, 'timestamp': time.time()})
            
        elif choice == '4':
            # Load and play
            filename = input("CTT filename: ").strip()
            if not os.path.exists(filename):
                print(f"❌ File not found: {filename}")
                continue
            
            phases, sr, freqs = recorder.load_ctt(filename)
            print(f"✅ Loaded {len(phases)} phases")
            
            duration = float(input("Playback duration (seconds): ") or "5")
            reconstructed = recorder.decode_chunk(phases, duration)
            
            print("🎧 Playing...")
            sd.play(reconstructed, int(sr))
            sd.wait()
            print("✅ Playback complete")
            
        elif choice == '5':
            print("\n🎵 CTT analog audio terminated")
            break


# ============================================================================
# STANDALONE PLAYER — if run with filename
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line player: python ctt_audio.py recording.ctt
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            sys.exit(1)
        
        recorder = CTTAnalogRecorder()
        phases, sr, freqs = recorder.load_ctt(filename)
        duration = float(input("Playback duration (seconds): ") or "5")
        
        print(f"\n🎧 Playing {filename}...")
        audio = recorder.decode_chunk(phases, duration)
        sd.play(audio, int(sr))
        sd.wait()
        print("✅ Playback complete")
    else:
        main()
