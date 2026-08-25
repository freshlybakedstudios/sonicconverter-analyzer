#include "Loudness.h"
#include "FeatureMath.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sonic {

// pyloudnorm Meter(44100) coefficients, printed from the installed package.
// high_shelf: G=4.0 dB, Q=0.70710678, fc=1500 Hz
static constexpr double SH_B[3] = { 1.5309095946396625, -2.651169032402396, 1.1691668584809876 };
static constexpr double SH_A[3] = { 1.0, -1.663750110244495, 0.7126575309627482 };
// high_pass: Q=0.5, fc=38 Hz
static constexpr double HP_B[3] = { 0.994607809439911, -1.989215618879822, 0.994607809439911 };
static constexpr double HP_A[3] = { 1.0, -1.9892010416922554, 0.9892301960673886 };

// scipy.signal.lfilter, direct form II transposed, zero initial state.
static void lfilterInPlace(const double b[3], const double a[3], std::vector<double>& x)
{
    double z1 = 0.0, z2 = 0.0;
    for (double& v : x) {
        const double in = v;
        const double out = b[0] * in + z1;
        z1 = b[1] * in - a[1] * out + z2;
        z2 = b[2] * in - a[2] * out;
        v = out;
    }
}

void kWeight(std::vector<double>& x)
{
    lfilterInPlace(SH_B, SH_A, x);
    lfilterInPlace(HP_B, HP_A, x);
}

// Shared gating core: z[j] mean-square blocks are already summed across
// channels (G=1 per channel for L/R/mono).
static double gatedLoudness(const std::vector<double>& z)
{
    if (z.empty())
        return -HUGE_VAL;

    std::vector<double> l(z.size());
    for (size_t j = 0; j < z.size(); ++j)
        l[j] = -0.691 + 10.0 * std::log10(z[j]); // log10(0) -> -inf, like numpy

    // Absolute gate (>= -70), then relative gate (strictly > both).
    double zSum = 0.0; int zCount = 0;
    for (size_t j = 0; j < z.size(); ++j)
        if (l[j] >= -70.0) { zSum += z[j]; ++zCount; }
    if (zCount == 0)
        return -HUGE_VAL;

    const double gammaR = -0.691 + 10.0 * std::log10(zSum / zCount) - 10.0;

    double zSum2 = 0.0; int zCount2 = 0;
    for (size_t j = 0; j < z.size(); ++j)
        if (l[j] > gammaR && l[j] > -70.0) { zSum2 += z[j]; ++zCount2; }
    if (zCount2 == 0)
        return -HUGE_VAL;

    return -0.691 + 10.0 * std::log10(zSum2 / zCount2);
}

// Blocks exactly as pyloudnorm indexes them: l = int(0.4*(j*0.25)*rate),
// u = int(0.4*(j*0.25 + 1)*rate), numBlocks = round((T-0.4)/0.1)+1.
static std::vector<double> meanSquareBlocks(const std::vector<double>& kw)
{
    const double rate = ANALYSIS_SR, Tg = 0.4, step = 0.25;
    const double T = (double) kw.size() / rate;
    const int numBlocks = (int) std::llround((T - Tg) / (Tg * step)) + 1;
    std::vector<double> z;
    z.reserve((size_t) std::max(numBlocks, 0));
    for (int j = 0; j < numBlocks; ++j) {
        const int lo = (int) (Tg * (j * step) * rate);
        const int hi = (int) (Tg * (j * step + 1) * rate);
        if (lo < 0 || hi > (int) kw.size() || hi <= lo)
            continue;
        double s = 0.0;
        for (int i = lo; i < hi; ++i)
            s += kw[(size_t) i] * kw[(size_t) i];
        z.push_back(s / (Tg * rate));
    }
    return z;
}

double integratedLoudnessMono(const std::vector<double>& x)
{
    std::vector<double> kw = x;
    kWeight(kw);
    return gatedLoudness(meanSquareBlocks(kw));
}

double loudnessRange(const std::vector<double>& mono)
{
    // audio_analyzer.py: block_size = 0.4*sr, hop block_size//2, each block
    // measured with a fresh Meter.integrated_loudness (its own filter state,
    // hence NOT streaming momentary — we replicate the fresh state per block).
    const int block = (int) (0.4 * ANALYSIS_SR); // 17640
    if ((int) mono.size() <= block)
        return 0.0;

    std::vector<double> lv;
    for (int i = 0; i + block <= (int) mono.size() && i < (int) mono.size() - block; i += block / 2) {
        std::vector<double> blk(mono.begin() + i, mono.begin() + i + block);
        const double bl = integratedLoudnessMono(blk);
        if (!std::isnan(bl)) // -inf passes, matching the Python append
            lv.push_back(bl);
    }
    if (lv.size() < 2)
        return 0.0;
    std::sort(lv.begin(), lv.end());
    return percentileSorted(lv, 95.0) - percentileSorted(lv, 10.0);
}

// --- streaming since-reset ---

IntegratedLoudnessStream::IntegratedLoudnessStream(int numChannels) : numCh(numChannels)
{
    reset();
}

void IntegratedLoudnessStream::reset()
{
    shelfState.assign((size_t) numCh, {});
    hpState.assign((size_t) numCh, {});
    pending.assign((size_t) numCh, {});
    blockPowerSum.clear();
}

void IntegratedLoudnessStream::push(const std::vector<const double*>& chans, int n)
{
    for (int c = 0; c < numCh; ++c) {
        auto& shs = shelfState[(size_t) c];
        auto& hps = hpState[(size_t) c];
        auto& buf = pending[(size_t) c];
        buf.reserve(buf.size() + (size_t) n);
        for (int i = 0; i < n; ++i) {
            double v = chans[(size_t) c][i];
            double out = SH_B[0] * v + shs.z1;
            shs.z1 = SH_B[1] * v - SH_A[1] * out + shs.z2;
            shs.z2 = SH_B[2] * v - SH_A[2] * out;
            v = out;
            out = HP_B[0] * v + hps.z1;
            hps.z1 = HP_B[1] * v - HP_A[1] * out + hps.z2;
            hps.z2 = HP_B[2] * v - HP_A[2] * out;
            buf.push_back(out);
        }
    }
    // Emit completed 400ms blocks (75% overlap -> hop 100ms)
    while ((int) pending[0].size() >= BLOCK) {
        double zSum = 0.0;
        for (int c = 0; c < numCh; ++c) {
            const auto& buf = pending[(size_t) c];
            double s = 0.0;
            for (int i = 0; i < BLOCK; ++i)
                s += buf[(size_t) i] * buf[(size_t) i];
            zSum += s / (double) BLOCK;
        }
        blockPowerSum.push_back(zSum);
        for (int c = 0; c < numCh; ++c)
            pending[(size_t) c].erase(pending[(size_t) c].begin(),
                                      pending[(size_t) c].begin() + HOP);
    }
    // Cap history (~20 min at 10 blocks/s) so a forgotten session can't grow unbounded
    constexpr size_t MAX_BLOCKS = 12000;
    if (blockPowerSum.size() > MAX_BLOCKS)
        blockPowerSum.erase(blockPowerSum.begin(),
                            blockPowerSum.begin() + (long) (blockPowerSum.size() - MAX_BLOCKS));
}

double IntegratedLoudnessStream::loudness() const
{
    return gatedLoudness(blockPowerSum);
}

} // namespace sonic
