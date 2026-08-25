#include "StftFeatures.h"
#include "FeatureMath.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include "kiss_fftr.h"

namespace sonic {

static constexpr int N_FFT = 2048;
static constexpr int HOP = 512;
static constexpr int BINS = N_FFT / 2 + 1;

StftResult computeStftFeatures(const std::vector<double>& mono)
{
    StftResult res;
    const int L = (int) mono.size();
    if (L < HOP)
        return res;

    // center=True: pad n_fft//2 zeros each side (librosa 0.11 pad_mode='constant')
    const int pad = N_FFT / 2;
    const int nFrames = 1 + L / HOP;

    // periodic Hann, matching scipy.signal.get_window('hann', 2048, fftbins=True)
    static const std::vector<double> hann = [] {
        std::vector<double> w((size_t) N_FFT);
        for (int i = 0; i < N_FFT; ++i)
            w[(size_t) i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double) i / (double) N_FFT);
        return w;
    }();

    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr);
    std::vector<double> frame((size_t) N_FFT);
    std::vector<kiss_fft_cpx> spec((size_t) BINS);

    // Magnitude spectrogram, kept whole because spectral_flux's top_db clip
    // is relative to the global max.
    std::vector<double> S((size_t) nFrames * BINS);

    std::vector<double> centroids((size_t) nFrames);
    std::vector<double> rolloffs((size_t) nFrames);
    std::vector<double> bandwidths((size_t) nFrames);

    for (int t = 0; t < nFrames; ++t) {
        const int start = t * HOP - pad; // in unpadded coordinates
        for (int i = 0; i < N_FFT; ++i) {
            const int idx = start + i;
            const double v = (idx >= 0 && idx < L) ? mono[(size_t) idx] : 0.0;
            frame[(size_t) i] = v * hann[(size_t) i];
        }
        kiss_fftr(cfg, frame.data(), spec.data());

        double total = 0.0, weighted = 0.0;
        double* Sf = &S[(size_t) t * BINS];
        for (int k = 0; k < BINS; ++k) {
            const double m = std::sqrt(spec[(size_t) k].r * spec[(size_t) k].r
                                     + spec[(size_t) k].i * spec[(size_t) k].i);
            Sf[k] = m;
            total += m;
            weighted += m * ((double) k * ANALYSIS_SR / (double) N_FFT);
        }

        const double centroid = total > 0.0 ? weighted / total : 0.0;
        centroids[(size_t) t] = centroid;

        // rolloff: lowest freq where cumulative magnitude >= 0.85 * total
        double cum = 0.0, roll = 0.0;
        if (total > 0.0) {
            const double thresh = 0.85 * total;
            for (int k = 0; k < BINS; ++k) {
                cum += Sf[k];
                if (cum >= thresh) { roll = (double) k * ANALYSIS_SR / (double) N_FFT; break; }
            }
        }
        rolloffs[(size_t) t] = roll;

        // bandwidth (norm=True, p=2): sqrt(sum(Snorm * (f - centroid)^2))
        double bw = 0.0;
        if (total > 0.0) {
            for (int k = 0; k < BINS; ++k) {
                const double d = (double) k * ANALYSIS_SR / (double) N_FFT - centroid;
                bw += (Sf[k] / total) * d * d;
            }
            bw = std::sqrt(bw);
        }
        bandwidths[(size_t) t] = bw;
    }
    kiss_fftr_free(cfg);

    auto mean = [](const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / (double) v.size();
    };

    res.brightness = mean(centroids);
    double var = 0.0;
    for (double c : centroids) var += (c - res.brightness) * (c - res.brightness);
    res.brightness_variance = centroids.empty() ? 0.0 : var / (double) centroids.size();
    res.spectral_rolloff = mean(rolloffs);
    res.spectral_complexity = mean(bandwidths) / ANALYSIS_SR;

    // spectral_flux: amplitude_to_db (amin=1e-5, ref=1, top_db=80 vs global
    // max), then sum-over-bins of frame-to-frame diff, mean of abs.
    {
        double dbMax = -HUGE_VAL;
        std::vector<double>& dB = S; // convert in place
        for (double& m : dB) {
            m = 20.0 * std::log10(std::max(m, 1e-5));
            dbMax = std::max(dbMax, m);
        }
        const double floorDb = dbMax - 80.0;
        for (double& m : dB)
            m = std::max(m, floorDb);

        double fluxSum = 0.0;
        for (int t = 1; t < nFrames; ++t) {
            double d = 0.0;
            const double* a = &dB[(size_t) (t - 1) * BINS];
            const double* b = &dB[(size_t) t * BINS];
            for (int k = 0; k < BINS; ++k)
                d += b[k] - a[k];
            fluxSum += std::abs(d);
        }
        res.spectral_flux = nFrames > 1 ? fluxSum / (double) (nFrames - 1) : 0.0;
    }

    // zcr: separate framing — librosa pads with mode='edge', counts sign
    // changes via signbit (|x| <= 1e-10 treated as 0, signbit(0)=positive);
    // zero_crossing_rate passes pad=False, so the frame's first sample never
    // counts as a crossing.
    {
        double zcrSum = 0.0;
        for (int t = 0; t < nFrames; ++t) {
            const int start = t * HOP - pad;
            int count = 0;
            bool prevSign = false;
            for (int i = 0; i < N_FFT; ++i) {
                int idx = start + i;
                idx = std::max(0, std::min(idx, L - 1)); // edge padding
                double v = mono[(size_t) idx];
                if (std::abs(v) <= 1e-10) v = 0.0;
                const bool sign = std::signbit(v);
                if (i > 0 && sign != prevSign) ++count;
                prevSign = sign;
            }
            zcrSum += (double) count / (double) N_FFT;
        }
        res.zcr = zcrSum / (double) nFrames;
    }

    return res;
}

} // namespace sonic
