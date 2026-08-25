#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include "FeatureMath.h"
#include "Loudness.h"

// Plugin-side glue: lock-free tap on the audio thread, background analysis
// thread that maintains a rolling 4 s window at 44.1 kHz and recomputes the
// full Python-parity feature set (~2 Hz). Also tracks "since reset" stereo
// integrated LUFS and true peak for engineering honesty (the windowed mono
// values are the target-comparable ones).

namespace sonic {

class FeatureEngine : private juce::Thread {
public:
    struct Snapshot {
        WindowFeatures window;
        // Loudest 4s window seen since the last pass reset — the same statistic
        // the analyzer measures on a full track, so it holds still and stays
        // target-comparable while a song plays through.
        WindowFeatures best;
        bool hasBest = false;
        double bestEnergy = 0.0;
        double sinceResetLufs = -HUGE_VAL;
        double sinceResetTruePeakDb = -200.0;
        bool windowFull = false;   // 4 s of audio accumulated
        bool receivingAudio = false;
        juce::int64 updateCount = 0;
    };

    FeatureEngine();
    ~FeatureEngine() override;

    void prepare(double sessionSampleRate);
    void release();

    // Audio thread. Mono sources may pass L == R.
    void push(const float* L, const float* R, int numSamples);

    Snapshot getSnapshot() const;
    void resetSinceReset();
    void resetPass(); // clear the held loudest-window measurement

private:
    void run() override;
    void drainFifoAndResample();
    void analyzeWindow();

    // FIFO: interleaved stereo floats written by the audio thread
    juce::AbstractFifo fifo { 1 << 16 };
    std::vector<float> fifoBuf;

    double sessionRate = 44100.0;
    juce::LagrangeInterpolator resampL, resampR;
    std::vector<float> resampleCarryL, resampleCarryR;
    std::vector<float> resampledL, resampledR;

    // Rolling window ring (44.1k domain)
    std::vector<double> ringL, ringR;
    int ringWrite = 0;
    juce::int64 ringTotal = 0;

    IntegratedLoudnessStream sinceReset { 2 };
    double sinceResetTp = -200.0;

    // Window copies reused across ticks (analysis thread only)
    std::vector<double> winMono, winL, winR;

    mutable juce::SpinLock snapshotLock;
    Snapshot latest;
    std::atomic<juce::int64> samplesSeen { 0 };
    juce::int64 lastSeen = 0;
    std::atomic<bool> resetRequested { false };
    std::atomic<bool> passResetRequested { false };
    WindowFeatures bestWindow;      // analysis thread only
    double bestWindowEnergy = 0.0;
    bool hasBestWindow = false;
};

} // namespace sonic
