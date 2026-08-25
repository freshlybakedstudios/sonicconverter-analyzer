#include "FeatureEngine.h"
#include "BandAnalyzer.h"
#include <cmath>

namespace sonic {

FeatureEngine::FeatureEngine() : juce::Thread("SonicMeter Analysis")
{
    fifoBuf.resize((size_t) fifo.getTotalSize() * 2);
    ringL.assign(WINDOW_SAMPLES, 0.0);
    ringR.assign(WINDOW_SAMPLES, 0.0);
    winMono.resize(WINDOW_SAMPLES);
    winL.resize(WINDOW_SAMPLES);
    winR.resize(WINDOW_SAMPLES);
}

FeatureEngine::~FeatureEngine()
{
    release();
}

void FeatureEngine::prepare(double sessionSampleRate)
{
    release();
    sessionRate = sessionSampleRate;
    fifo.reset();
    resampL.reset();
    resampR.reset();
    resampleCarryL.clear();
    resampleCarryR.clear();
    ringL.assign(WINDOW_SAMPLES, 0.0);
    ringR.assign(WINDOW_SAMPLES, 0.0);
    ringWrite = 0;
    ringTotal = 0;
    sinceReset.reset();
    sinceResetTp = -200.0;
    bestWindowEnergy = 0.0;
    hasBestWindow = false;
    startThread(juce::Thread::Priority::low);
}

void FeatureEngine::release()
{
    stopThread(2000);
}

void FeatureEngine::push(const float* L, const float* R, int numSamples)
{
    // Audio thread: interleave into the FIFO. If the analysis thread stalls,
    // drop the oldest by skipping — never block.
    int start1, size1, start2, size2;
    fifo.prepareToWrite(numSamples, start1, size1, start2, size2);
    int i = 0;
    for (int s = start1; s < start1 + size1; ++s, ++i) {
        fifoBuf[(size_t) s * 2] = L[i];
        fifoBuf[(size_t) s * 2 + 1] = R[i];
    }
    for (int s = start2; s < start2 + size2; ++s, ++i) {
        fifoBuf[(size_t) s * 2] = L[i];
        fifoBuf[(size_t) s * 2 + 1] = R[i];
    }
    fifo.finishedWrite(size1 + size2);
    samplesSeen.fetch_add(numSamples, std::memory_order_relaxed);
    notify();
}

void FeatureEngine::drainFifoAndResample()
{
    const int ready = fifo.getNumReady();
    if (ready <= 0)
        return;

    int start1, size1, start2, size2;
    fifo.prepareToRead(ready, start1, size1, start2, size2);
    auto appendRange = [this](int start, int size) {
        for (int s = start; s < start + size; ++s) {
            resampleCarryL.push_back(fifoBuf[(size_t) s * 2]);
            resampleCarryR.push_back(fifoBuf[(size_t) s * 2 + 1]);
        }
    };
    appendRange(start1, size1);
    appendRange(start2, size2);
    fifo.finishedRead(size1 + size2);

    const double ratio = sessionRate / ANALYSIS_SR;
    int produced;
    if (std::abs(ratio - 1.0) < 1e-9) {
        produced = (int) resampleCarryL.size();
        resampledL.assign(resampleCarryL.begin(), resampleCarryL.end());
        resampledR.assign(resampleCarryR.begin(), resampleCarryR.end());
        resampleCarryL.clear();
        resampleCarryR.clear();
    } else {
        // Keep a small guard so the interpolator never reads past the input.
        const int avail = (int) resampleCarryL.size();
        produced = (int) std::floor((avail - 8) / ratio);
        if (produced <= 0)
            return;
        resampledL.resize((size_t) produced);
        resampledR.resize((size_t) produced);
        const int usedL = resampL.process(ratio, resampleCarryL.data(), resampledL.data(), produced);
        const int usedR = resampR.process(ratio, resampleCarryR.data(), resampledR.data(), produced);
        resampleCarryL.erase(resampleCarryL.begin(), resampleCarryL.begin() + usedL);
        resampleCarryR.erase(resampleCarryR.begin(), resampleCarryR.begin() + usedR);
    }

    // Append to ring + feed since-reset loudness
    std::vector<double> dl((size_t) produced), dr((size_t) produced);
    for (int i = 0; i < produced; ++i) {
        dl[(size_t) i] = resampledL[(size_t) i];
        dr[(size_t) i] = resampledR[(size_t) i];
        ringL[(size_t) ringWrite] = dl[(size_t) i];
        ringR[(size_t) ringWrite] = dr[(size_t) i];
        ringWrite = (ringWrite + 1) % WINDOW_SAMPLES;
    }
    ringTotal += produced;
    sinceReset.push({ dl.data(), dr.data() }, produced);
}

void FeatureEngine::analyzeWindow()
{
    const bool full = ringTotal >= WINDOW_SAMPLES;
    Snapshot snap;
    snap.windowFull = full;

    const juce::int64 seen = samplesSeen.load(std::memory_order_relaxed);
    snap.receivingAudio = seen != lastSeen;
    lastSeen = seen;

    if (full) {
        // Unroll ring into chronological windows
        for (int i = 0; i < WINDOW_SAMPLES; ++i) {
            const int idx = (ringWrite + i) % WINDOW_SAMPLES;
            winL[(size_t) i] = ringL[(size_t) idx];
            winR[(size_t) i] = ringR[(size_t) idx];
            // librosa mono averages in float32 before upcasting
            winMono[(size_t) i] = (double) (float) ((winL[(size_t) i] + winR[(size_t) i]) * 0.5);
        }
        snap.window = computeWindowFeatures(winMono, winL, winR);

        // Track the loudest window of the pass — the statistic the analyzer
        // measures on a full track, so it can be held steady in the UI.
        if (snap.window.energy > bestWindowEnergy) {
            bestWindowEnergy = snap.window.energy;
            bestWindow = snap.window;
            hasBestWindow = true;
        }

        // Per-channel windowed TP maxed since reset (windows at ~500 ms hop
        // overlap 8x, so every sample is covered)
        auto& ba = sharedBandAnalyzer(WINDOW_SAMPLES);
        sinceResetTp = std::max({ sinceResetTp, ba.truePeakDb(winL), ba.truePeakDb(winR) });
    }

    snap.sinceResetLufs = sinceReset.loudness();
    snap.sinceResetTruePeakDb = sinceResetTp;
    snap.best = bestWindow;
    snap.hasBest = hasBestWindow;
    snap.bestEnergy = bestWindowEnergy;

    const juce::SpinLock::ScopedLockType sl(snapshotLock);
    snap.updateCount = latest.updateCount + 1;
    if (!full)
        snap.window = latest.window; // keep last good values while refilling
    latest = snap;
}

void FeatureEngine::run()
{
    juce::int64 lastAnalysis = 0;
    while (!threadShouldExit()) {
        wait(50);
        if (threadShouldExit())
            break;
        if (resetRequested.exchange(false)) {
            sinceReset.reset();
            sinceResetTp = -200.0;
        }
        if (passResetRequested.exchange(false)) {
            bestWindowEnergy = 0.0;
            hasBestWindow = false;
        }
        drainFifoAndResample();
        const juce::int64 now = juce::Time::getMillisecondCounter();
        if (now - lastAnalysis >= 500) {
            lastAnalysis = now;
            analyzeWindow();
        }
    }
}

FeatureEngine::Snapshot FeatureEngine::getSnapshot() const
{
    const juce::SpinLock::ScopedLockType sl(snapshotLock);
    return latest;
}

void FeatureEngine::resetSinceReset()
{
    // The analysis thread owns the accumulators — hand the reset off to it.
    resetRequested.store(true);
    notify();
}

void FeatureEngine::resetPass()
{
    passResetRequested.store(true);
    notify();
}

} // namespace sonic
