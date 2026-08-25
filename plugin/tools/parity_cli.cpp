// parity_cli <file.wav> — computes the shared engine's window features over
// the WHOLE file (the Python harness pre-cuts the exact 4 s segment) and
// prints them as JSON. Input must already be 44.1 kHz; we refuse otherwise so
// no resampler ever sits between the two implementations under test.

#include <juce_audio_formats/juce_audio_formats.h>
#include "engine/FeatureMath.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: parity_cli <file.wav>\n";
        return 2;
    }

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(
        fm.createReaderFor(juce::File(juce::String(argv[1]))));
    if (reader == nullptr) {
        std::cerr << "cannot read " << argv[1] << "\n";
        return 2;
    }
    if ((int) reader->sampleRate != (int) sonic::ANALYSIS_SR) {
        std::cerr << "expected 44100 Hz input, got " << reader->sampleRate << "\n";
        return 2;
    }

    const int numCh = (int) reader->numChannels;
    const int n = (int) reader->lengthInSamples;
    juce::AudioBuffer<float> buf(numCh, n);
    reader->read(&buf, 0, n, 0, true, true);

    std::vector<double> mono((size_t) n), left, right;
    if (numCh >= 2) {
        left.resize((size_t) n);
        right.resize((size_t) n);
        const float* L = buf.getReadPointer(0);
        const float* R = buf.getReadPointer(1);
        for (int i = 0; i < n; ++i) {
            left[(size_t) i] = L[i];
            right[(size_t) i] = R[i];
            // librosa mono: mean across channels in float32, then upcast
            mono[(size_t) i] = (double) ((L[i] + R[i]) * 0.5f);
        }
    } else {
        const float* M = buf.getReadPointer(0);
        for (int i = 0; i < n; ++i)
            mono[(size_t) i] = M[i];
    }

    const sonic::WindowFeatures f = sonic::computeWindowFeatures(mono, left, right);

    std::cout << "{";
    bool first = true;
    for (const auto& [k, v] : f.asMap()) {
        if (!first) std::cout << ",";
        first = false;
        std::cout << "\"" << k << "\":";
        if (std::isfinite(v)) std::cout << juce::String(v, 10).toStdString();
        else std::cout << (v < 0 ? "-1e308" : "1e308"); // JSON-safe infinity
    }
    std::cout << "}\n";
    return 0;
}
