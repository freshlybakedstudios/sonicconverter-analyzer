#pragma once
#include <vector>

// librosa-default STFT features (n_fft=2048, hop=512, periodic Hann,
// center=True with zero padding — librosa 0.11 pad_mode='constant').
// Mirrors audio_analyzer.py lines 712-724 and 792-797.

namespace sonic {

struct StftResult {
    double brightness = 0.0;          // mean spectral centroid (Hz)
    double brightness_variance = 0.0; // np.var (ddof=0) of centroid frames
    double spectral_rolloff = 0.0;    // mean rolloff at 0.85 (Hz)
    double spectral_complexity = 0.0; // mean spectral bandwidth / sr
    double zcr = 0.0;                 // mean zero-crossing rate
    double spectral_flux = 0.0;       // mean |sum-over-bins dB frame diff|
};

StftResult computeStftFeatures(const std::vector<double>& mono);

} // namespace sonic
