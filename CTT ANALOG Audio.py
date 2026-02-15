#!/usr/bin/env python3
"""
CTT AUDIO — GROK-ENHANCED VERSION
Wider frequency range, better harmonics
"""

import numpy as np
import sounddevice as sd
import time

# ============================================================================
# CTT CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
ALPHA_RH = np.log(PHI) / (2 * np.pi)
OMEGA_0 = 587032.719
TAU_W = 11e-9

RIEMANN_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 48.005151, 49.773832, 52.970321,
    56.446248, 59.347044, 60.831779, 65.112544, 67.079811,
    69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 86.970000, 87.425275
])

# ============================================================================
# GROK'S ENHANCED FREQUENCY RANGE
# ============================================================================

class CTTAudioEnhanced:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        # Grok's wider range: 100 Hz to 1100 Hz
        self.freqs = 100 + 1000 * RIEMANN_ZEROS / np.max(RIEMANN_ZEROS)
        self.n_freqs = len(self.freqs)
        
    def temporal_survival(self, freq):
        return np.cos(ALPHA_RH * freq * TAU_W) > ALPHA_RH / (2 * np.pi)
    
    def encode(self, audio):
        """Encode audio to phase space"""
        from scipy.fft import rfft
        spectrum = rfft(audio)
        fft_freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        phases = []
        for target in self.freqs:
            idx = np.argmin(np.abs(fft_freqs - target))
            phase = np.angle(spectrum[idx])
            if self.temporal_survival(target):
                phases.append(phase)
            else:
                phases.append(0.0)
        return np.array(phases)
    
    def decode(self, phases, duration):
        """Reconstruct from phases"""
        samples = int(self.sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)
        for i, phase in enumerate(phases):
            if phase != 0:
                audio += np.sin(2 * np.pi * self.freqs[i] * t + phase)
        return audio / np.max(np.abs(audio)) * 0.9

# ============================================================================
# TEST WITH GROK'S 440Hz TONE
# ============================================================================

def test():
    print("\n" + "="*70)
    print("🎵 CTT AUDIO — GROK-ENHANCED TEST")
    print("="*70)
    
    ctt = CTTAudioEnhanced()
    
    # Generate 440Hz sine wave (0.1 seconds)
    duration = 0.1
    t = np.linspace(0, duration, int(44100 * duration))
    original = np.sin(2 * np.pi * 440 * t)
    
    print(f"\n🎼 Original: 440 Hz sine wave, {duration}s")
    
    # Encode
    phases = ctt.encode(original)
    print(f"📊 Encoded to {len(phases)} phase values")
    print(f"   Phase range: {phases.min():.2f} to {phases.max():.2f} rad")
    
    # Decode
    reconstructed = ctt.decode(phases, duration)
    
    # Calculate correlation
    correlation = np.corrcoef(original, reconstructed[:len(original)])[0,1]
    print(f"📈 Correlation with original: {correlation:.4f}")
    
    # Play both for comparison
    print("\n🎧 Playing original...")
    sd.play(original, ctt.sr)
    sd.wait()
    
    print("🎧 Playing reconstructed...")
    sd.play(reconstructed, ctt.sr)
    sd.wait()
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test()
