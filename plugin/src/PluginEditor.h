#pragma once
#include "PluginProcessor.h"
#include "ui/MeterRowComponent.h"

class SonicMeterEditor : public juce::AudioProcessorEditor,
                         private juce::Timer,
                         public juce::FileDragAndDropTarget {
public:
    explicit SonicMeterEditor(SonicMeterProcessor&);
    ~SonicMeterEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

private:
    void timerCallback() override;
    void rebuildRows();
    void loadTargetsFromFile(const juce::File& f);
    void chooseTargetsFile();

    SonicMeterProcessor& proc;
    int seenTargetsVersion = -1;

    juce::TextButton loadButton { "Load targets..." };
    juce::TextButton resetButton { "Reset peaks" };
    juce::TextButton modeButton { "Mode: Hold loudest" };
    juce::TextButton passButton { "New pass" };
    juce::Label headerLabel;
    juce::Label statusLabel;
    bool holdMode = true; // hold the loudest 4s window of the pass (analyzer statistic)

    struct DomainHeader : juce::Component {
        juce::String title;
        void paint(juce::Graphics& g) override;
    };

    juce::Viewport viewport;
    juce::Component rowContainer;
    std::vector<std::unique_ptr<sonic::MeterRowComponent>> rows;
    std::vector<juce::String> rowFeatures; // parallel to rows
    std::vector<std::unique_ptr<DomainHeader>> domainHeaders;

    std::unique_ptr<juce::FileChooser> chooser;
    bool dragHover = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SonicMeterEditor)
};
