#include "BandAnalyzer.h"
#include "FeatureMath.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include "kiss_fftr.h"

namespace sonic {

static const double BAND_EDGES[7][2] = {
    { 20, 60 }, { 60, 250 }, { 250, 500 }, { 500, 2000 },
    { 2000, 4000 }, { 4000, 8000 }, { 8000, 20000 },
};

struct BandAnalyzer::Impl {
    int n = 0;
    kiss_fftr_cfg fwd = nullptr;   // n-point forward
    kiss_fftr_cfg inv4 = nullptr;  // 4n-point inverse (true peak)
    std::vector<kiss_fft_cpx> spec;     // n/2+1
    std::vector<kiss_fft_cpx> spec4;    // 2n+1 (= 4n/2+1)
    std::vector<double> mag;            // n/2+1
    std::vector<double> upsampled;      // 4n
};

BandAnalyzer::BandAnalyzer(int n) : impl(new Impl)
{
    impl->n = n;
    impl->fwd = kiss_fftr_alloc(n, 0, nullptr, nullptr);
    impl->inv4 = kiss_fftr_alloc(4 * n, 1, nullptr, nullptr);
    impl->spec.resize((size_t) n / 2 + 1);
    impl->spec4.resize((size_t) (2 * n) + 1);
    impl->mag.resize((size_t) n / 2 + 1);
    impl->upsampled.resize((size_t) (4 * n));
}

BandAnalyzer::~BandAnalyzer()
{
    kiss_fftr_free(impl->fwd);
    kiss_fftr_free(impl->inv4);
    delete impl;
}

// scipy.signal.resample(x, 4n) for real even-length input: copy positive-freq
// bins, split the (real) Nyquist bin in half, zero the rest, inverse at 4n,
// scale so amplitude is preserved. kiss_fftri is unnormalized, so the net
// scale is 1/n (irfft's 1/(4n) times scipy's amplitude factor 4).
static double upsampledPeak(BandAnalyzer::Impl* im)
{
    const int n = im->n;
    const int nyq = n / 2; // n even
    std::memset(im->spec4.data(), 0, im->spec4.size() * sizeof(kiss_fft_cpx));
    for (int k = 0; k <= nyq; ++k)
        im->spec4[(size_t) k] = im->spec[(size_t) k];
    im->spec4[(size_t) nyq].r *= 0.5;
    im->spec4[(size_t) nyq].i *= 0.5;

    kiss_fftri(im->inv4, im->spec4.data(), im->upsampled.data());

    double peak = 0.0;
    for (double v : im->upsampled)
        peak = std::max(peak, std::abs(v));
    return peak / (double) n;
}

BandResult BandAnalyzer::analyze(const std::vector<double>& x)
{
    BandResult r;
    const int n = impl->n;
    const int bins = n / 2 + 1;

    kiss_fftr(impl->fwd, x.data(), impl->spec.data());
    double total = 0.0;
    for (int k = 0; k < bins; ++k) {
        const auto& c = impl->spec[(size_t) k];
        impl->mag[(size_t) k] = std::sqrt(c.r * c.r + c.i * c.i);
        total += impl->mag[(size_t) k];
    }

    if (total > 0.0) {
        // freqs[k] = k * sr / n; mask: freq >= lo && freq < hi
        for (int b = 0; b < 7; ++b) {
            const double lo = BAND_EDGES[b][0], hi = BAND_EDGES[b][1];
            double s = 0.0;
            for (int k = 0; k < bins; ++k) {
                const double f = (double) k * ANALYSIS_SR / (double) n;
                if (f >= lo && f < hi)
                    s += impl->mag[(size_t) k];
            }
            r.ratios[b] = s / total;
        }

        // THD approximation: fundamental = argmax over bins[1:], harmonics 2-5
        int fund = 1;
        for (int k = 2; k < bins; ++k)
            if (impl->mag[(size_t) k] > impl->mag[(size_t) fund])
                fund = k;
        const double fundMag = impl->mag[(size_t) fund];
        double harm = 0.0;
        for (int h = 2; h <= 5; ++h) {
            const long idx = (long) fund * h;
            if (idx < bins)
                harm += impl->mag[(size_t) idx];
        }
        r.harmonic_distortion = harm / std::max(fundMag, 1e-10);
    }

    // True peak reuses the spectrum we just computed
    const double peak = upsampledPeak(impl);
    r.true_peak_dbfs = 20.0 * std::log10(std::max(peak, 1e-10));
    return r;
}

double BandAnalyzer::truePeakDb(const std::vector<double>& x)
{
    kiss_fftr(impl->fwd, x.data(), impl->spec.data());
    const double peak = upsampledPeak(impl);
    return 20.0 * std::log10(std::max(peak, 1e-10));
}

} // namespace sonic
