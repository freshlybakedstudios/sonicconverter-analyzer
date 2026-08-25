#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "engine/FeatureEngine.h"
#include "model/Targets.h"

class SonicMeterProcessor : public juce::AudioProcessor {
public:
    SonicMeterProcessor();
    ~SonicMeterProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "SonicMeter"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Targets are read/written from the message thread only.
    sonic::TargetsFile targets;
    std::atomic<int> targetsVersion { 0 }; // bump so the editor re-lays out

    sonic::FeatureEngine engine;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SonicMeterProcessor)
};
