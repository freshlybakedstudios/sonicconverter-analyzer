#include "FeatureMath.h"
#include "Loudness.h"
#include "StftFeatures.h"
#include "BandAnalyzer.h"
#include <cmath>
#include <algorithm>
#include <memory>

namespace sonic {

BandAnalyzer& sharedBandAnalyzer(int n)
{
    static thread_local std::unique_ptr<BandAnalyzer> cached;
    static thread_local int cachedN = 0;
    if (n != cachedN) {
        cached = std::make_unique<BandAnalyzer>(n);
        cachedN = n;
    }
    return *cached;
}

double percentileSorted(const std::vector<double>& sorted, double p)
{
    if (sorted.empty())
        return 0.0;
    const size_t n = sorted.size();
    const double idx = (p / 100.0) * (double) (n - 1);
    const size_t lo = (size_t) idx;
    const size_t hi = std::min(lo + 1, n - 1);
    const double frac = idx - (double) lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

WindowFeatures computeWindowFeatures(const std::vector<double>& mono,
                                     const std::vector<double>& left,
                                     const std::vector<double>& right)
{
    WindowFeatures f;
    const size_t n = mono.size();
    if (n == 0)
        return f;

    // --- Stereo trio (Python defaults kept when no stereo buffer) ---
    if (!left.empty() && left.size() == right.size()) {
        double midSq = 0.0, sideSq = 0.0;
        double sumL = 0.0, sumR = 0.0;
        for (size_t i = 0; i < left.size(); ++i) {
            const double m = (left[i] + right[i]) / 2.0;
            const double s = (left[i] - right[i]) / 2.0;
            midSq += m * m;
            sideSq += s * s;
            sumL += left[i];
            sumR += right[i];
        }
        const double midE = std::sqrt(midSq / (double) left.size());
        const double sideE = std::sqrt(sideSq / (double) left.size());
        f.stereo_width = sideE / std::max(midE, 1e-10);
        f.mid_side_ratio = midE / std::max(midE + sideE, 1e-10);

        // np.corrcoef(L, R): Pearson correlation
        const double meanL = sumL / (double) left.size();
        const double meanR = sumR / (double) right.size();
        double cov = 0.0, varL = 0.0, varR = 0.0;
        for (size_t i = 0; i < left.size(); ++i) {
            const double dl = left[i] - meanL, dr = right[i] - meanR;
            cov += dl * dr;
            varL += dl * dl;
            varR += dr * dr;
        }
        const double denom = std::sqrt(varL * varR);
        const double corr = denom > 0.0 ? cov / denom : NAN;
        f.stereo_correlation = std::isfinite(corr) ? corr : 1.0;
    }

    // --- Loudness ---
    {
        const double lufs = integratedLoudnessMono(mono);
        f.lufs_integrated = std::isnan(lufs) ? -30.0 : lufs;
        f.loudness_range = loudnessRange(mono);
    }

    // --- Whole-window FFT: bands, THD, true peak ---
    {
        const BandResult br = sharedBandAnalyzer((int) n).analyze(mono);
        f.sub_ratio = br.ratios[0];
        f.bass_ratio = br.ratios[1];
        f.low_mid_ratio = br.ratios[2];
        f.mid_ratio = br.ratios[3];
        f.high_mid_ratio = br.ratios[4];
        f.presence_ratio = br.ratios[5];
        f.air_ratio = br.ratios[6];
        f.harmonic_distortion = br.harmonic_distortion;
        f.true_peak_dbfs = br.true_peak_dbfs;
    }

    // --- Energy & dynamics ---
    {
        double sq = 0.0, peak = 0.0;
        for (double v : mono) {
            sq += v * v;
            peak = std::max(peak, std::abs(v));
        }
        f.energy = std::sqrt(sq / (double) n);

        std::vector<double> absSorted(n);
        for (size_t i = 0; i < n; ++i)
            absSorted[i] = std::abs(mono[i]);
        std::sort(absSorted.begin(), absSorted.end());
        const double p95 = percentileSorted(absSorted, 95.0);
        const double p10 = percentileSorted(absSorted, 10.0);
        f.dynamic_range = p10 > 1e-10
            ? std::min(20.0 * std::log10(p95 / p10), 60.0) : 10.0;

        f.crest_factor = f.energy > 0.0
            ? 20.0 * std::log10(peak / std::max(f.energy, 1e-10)) : 1.0;
        f.compression_amount = std::max(0.0, 1.0 - f.crest_factor / 26.0);
    }

    // --- STFT features ---
    {
        const StftResult sr = computeStftFeatures(mono);
        f.brightness = sr.brightness;
        f.brightness_variance = sr.brightness_variance;
        f.spectral_rolloff = sr.spectral_rolloff;
        f.spectral_complexity = sr.spectral_complexity;
        f.zcr = sr.zcr;
        f.spectral_flux = sr.spectral_flux;
        f.dissonance = sr.zcr * sr.spectral_flux / 1000.0;
    }

    // Python cleans NaN -> 0.0 at the end of _extract_core
    auto clean = [](double& v) { if (std::isnan(v)) v = 0.0; };
    for (double* p : { &f.stereo_width, &f.mid_side_ratio, &f.stereo_correlation,
                       &f.lufs_integrated, &f.loudness_range, &f.sub_ratio,
                       &f.bass_ratio, &f.low_mid_ratio, &f.mid_ratio,
                       &f.high_mid_ratio, &f.presence_ratio, &f.air_ratio,
                       &f.harmonic_distortion, &f.energy, &f.true_peak_dbfs,
                       &f.dynamic_range, &f.crest_factor, &f.compression_amount,
                       &f.brightness, &f.brightness_variance, &f.spectral_rolloff,
                       &f.spectral_complexity, &f.zcr, &f.spectral_flux,
                       &f.dissonance })
        clean(*p);

    return f;
}

std::map<std::string, double> WindowFeatures::asMap() const
{
    return {
        { "stereo_width", stereo_width },
        { "mid_side_ratio", mid_side_ratio },
        { "stereo_correlation", stereo_correlation },
        { "lufs_integrated", lufs_integrated },
        { "loudness_range", loudness_range },
        { "sub_ratio", sub_ratio },
        { "bass_ratio", bass_ratio },
        { "low_mid_ratio", low_mid_ratio },
        { "mid_ratio", mid_ratio },
        { "high_mid_ratio", high_mid_ratio },
        { "presence_ratio", presence_ratio },
        { "air_ratio", air_ratio },
        { "harmonic_distortion", harmonic_distortion },
        { "energy", energy },
        { "true_peak_dbfs", true_peak_dbfs },
        { "dynamic_range", dynamic_range },
        { "crest_factor", crest_factor },
        { "compression_amount", compression_amount },
        { "brightness", brightness },
        { "brightness_variance", brightness_variance },
        { "spectral_rolloff", spectral_rolloff },
        { "spectral_complexity", spectral_complexity },
        { "zcr", zcr },
        { "spectral_flux", spectral_flux },
        { "dissonance", dissonance },
    };
}

} // namespace sonic
