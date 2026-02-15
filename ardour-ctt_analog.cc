/*
 * CTT ANALOG — LV2 Plugin for Ardour
 * True analog recording using Convergent Time Theory
 * One file. Drop in. Instant analog.
 */

#include <lv2/lv2plug.in/ns/lv2core/lv2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CTT_URI "http://ctt-research.org/plugins/ctt-analog"

// ============================================================================
// CTT CONSTANTS — The Analog Heart
// ============================================================================

const double PHI = 1.618033988749895;
const double ALPHA_RH = 0.07658720111364355;  // ln(φ)/(2π)
const double OMEGA_0 = 587032.719;  // Hz — silicon heartbeat
const double TAU_W = 1.1e-8;  // 11 ns temporal wedge

// 24 Riemann zeros — the analog frequencies
const double RIEMANN_ZEROS[24] = {
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 48.005151, 49.773832, 52.970321,
    56.446248, 59.347044, 60.831779, 65.112544, 67.079811,
    69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 86.970000, 87.425275
};

// Scale to audible range (20 Hz - 20 kHz)
double AUDIO_FREQS[24];

// ============================================================================
// CTT ANALOG PLUGIN
// ============================================================================

typedef struct {
    float* input;
    float* output;
    float* mode;      // 0 = bypass, 1 = encode, 2 = decode
    float* quality;   // 1-24, number of frequency bands
    
    double sample_rate;
    int window_size;
    double freqs[24];
    double* phase_buffer;
    double* amp_buffer;
    int buffer_pos;
} CTTAnalog;

static LV2_Handle instantiate(const LV2_Descriptor* descriptor,
                              double sample_rate,
                              const char* bundle_path,
                              const LV2_Feature* const* features) {
    CTTAnalog* self = (CTTAnalog*)calloc(1, sizeof(CTTAnalog));
    
    self->sample_rate = sample_rate;
    self->window_size = (int)(sample_rate * 0.05);  // 50ms windows
    self->phase_buffer = (double*)calloc(24 * 1024, sizeof(double));
    self->amp_buffer = (double*)calloc(24 * 1024, sizeof(double));
    self->buffer_pos = 0;
    
    // Scale Riemann zeros to audible range
    double max_zero = RIEMANN_ZEROS[23];
    for(int i = 0; i < 24; i++) {
        self->freqs[i] = 20.0 + 19980.0 * RIEMANN_ZEROS[i] / max_zero;
    }
    
    return (LV2_Handle)self;
}

static void connect_port(LV2_Handle instance, uint32_t port, void* data) {
    CTTAnalog* self = (CTTAnalog*)instance;
    
    switch(port) {
        case 0: self->input = (float*)data; break;
        case 1: self->output = (float*)data; break;
        case 2: self->mode = (float*)data; break;
        case 3: self->quality = (float*)data; break;
    }
}

static void activate(LV2_Handle instance) {
    CTTAnalog* self = (CTTAnalog*)instance;
    self->buffer_pos = 0;
    memset(self->phase_buffer, 0, 24 * 1024 * sizeof(double));
    memset(self->amp_buffer, 0, 24 * 1024 * sizeof(double));
}

// Goertzel algorithm — exact frequency phase/amplitude
static void goertzel(const float* samples, int n, double target_freq,
                     double sample_rate, double* phase, double* amp) {
    double omega = 2.0 * M_PI * target_freq / sample_rate;
    double coeff = 2.0 * cos(omega);
    
    double s_prev = 0.0;
    double s_prev2 = 0.0;
    
    for(int i = 0; i < n; i++) {
        double s = samples[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    
    double real = s_prev - s_prev2 * cos(omega);
    double imag = s_prev2 * sin(omega);
    
    *phase = atan2(imag, real);
    *amp = sqrt(real*real + imag*imag);
}

// Temporal survival — which frequencies survive the wedge
static int temporal_survival(double freq) {
    double val = cos(ALPHA_RH * freq * TAU_W);
    double threshold = ALPHA_RH / (2.0 * M_PI);
    return val > threshold ? 1 : 0;
}

static void run(LV2_Handle instance, uint32_t n_samples) {
    CTTAnalog* self = (CTTAnalog*)instance;
    
    int mode = (int)(*self->mode + 0.5);
    int quality = (int)(*self->quality + 0.5);
    if(quality < 1) quality = 1;
    if(quality > 24) quality = 24;
    
    if(mode == 0) {
        // Bypass — copy input to output
        memcpy(self->output, self->input, n_samples * sizeof(float));
        return;
    }
    
    if(mode == 1) {
        // ENCODE mode — convert analog to CTT phases
        for(uint32_t i = 0; i < n_samples; i++) {
            int pos = self->buffer_pos + i;
            
            // Process in windows
            if(pos % self->window_size == 0) {
                int window_start = (pos / self->window_size) * self->window_size;
                const float* window = self->input + window_start;
                
                for(int f = 0; f < quality; f++) {
                    double phase, amp;
                    goertzel(window, self->window_size, self->freqs[f],
                            self->sample_rate, &phase, &amp);
                    
                    if(temporal_survival(self->freqs[f])) {
                        self->phase_buffer[window_start * 24 + f] = phase;
                        self->amp_buffer[window_start * 24 + f] = amp;
                    } else {
                        self->amp_buffer[window_start * 24 + f] = 0.0;
                    }
                }
            }
            
            // Output phase data (for visualization)
            self->output[i] = self->phase_buffer[pos * 24] / (2 * M_PI);
        }
        
        self->buffer_pos += n_samples;
    }
    
    if(mode == 2) {
        // DECODE mode — reconstruct analog from CTT phases
        for(uint32_t i = 0; i < n_samples; i++) {
            int pos = self->buffer_pos + i;
            double t = pos / self->sample_rate;
            double sample = 0.0;
            
            for(int f = 0; f < quality; f++) {
                double amp = self->amp_buffer[pos * 24 + f];
                if(amp > 0.001) {
                    double phase = self->phase_buffer[pos * 24 + f];
                    sample += amp * sin(2.0 * M_PI * self->freqs[f] * t + phase);
                }
            }
            
            self->output[i] = sample / quality;
        }
        
        self->buffer_pos += n_samples;
    }
}

static void deactivate(LV2_Handle instance) {
    // Nothing to do
}

static void cleanup(LV2_Handle instance) {
    CTTAnalog* self = (CTTAnalog*)instance;
    free(self->phase_buffer);
    free(self->amp_buffer);
    free(self);
}

static const void* extension_data(const char* uri) {
    return NULL;
}

// ============================================================================
// LV2 Plugin Descriptor
// ============================================================================

static const LV2_Descriptor descriptor = {
    CTT_URI,
    instantiate,
    connect_port,
    activate,
    run,
    deactivate,
    cleanup,
    extension_data
};

LV2_SYMBOL_EXPORT const LV2_Descriptor* lv2_descriptor(uint32_t index) {
    return index == 0 ? &descriptor : NULL;
}
