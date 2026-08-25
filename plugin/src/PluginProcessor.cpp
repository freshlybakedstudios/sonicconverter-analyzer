#include "PluginProcessor.h"
#include "PluginEditor.h"

SonicMeterProcessor::SonicMeterProcessor()
    : juce::AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

void SonicMeterProcessor::prepareToPlay(double sampleRate, int)
{
    engine.prepare(sampleRate);
}

void SonicMeterProcessor::releaseResources()
{
    engine.release();
}

bool SonicMeterProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto in = layouts.getMainInputChannelSet();
    const auto out = layouts.getMainOutputChannelSet();
    return in == out
        && (in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo());
}

void SonicMeterProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    // Pure passthrough — we only tap.
    const float* L = buffer.getReadPointer(0);
    const float* R = buffer.getNumChannels() > 1 ? buffer.getReadPointer(1) : L;
    engine.push(L, R, buffer.getNumSamples());
}

juce::AudioProcessorEditor* SonicMeterProcessor::createEditor()
{
    return new SonicMeterEditor(*this);
}

void SonicMeterProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    // Persist the raw targets JSON so a reopened Pro Tools session comes back
    // with its targets loaded.
    juce::ValueTree state("SonicMeterState");
    state.setProperty("targetsJson", targets.rawJson, nullptr);
    juce::MemoryOutputStream mos(destData, false);
    state.writeToStream(mos);
}

void SonicMeterProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    const auto state = juce::ValueTree::readFromData(data, (size_t) sizeInBytes);
    if (!state.isValid())
        return;
    const juce::String json = state.getProperty("targetsJson").toString();
    if (json.isNotEmpty() && targets.loadFromJson(json))
        targetsVersion.fetch_add(1);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SonicMeterProcessor();
}
