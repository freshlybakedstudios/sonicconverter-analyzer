#include "PluginEditor.h"

using sonic::theme::accent;
using sonic::theme::accentLight;
using sonic::theme::bg;
using sonic::theme::border;
using sonic::theme::card;
using sonic::theme::surface;
using sonic::theme::text;
using sonic::theme::textDim;

SonicMeterEditor::SonicMeterEditor(SonicMeterProcessor& p)
    : juce::AudioProcessorEditor(p), proc(p)
{
    setResizable(true, true);
    setResizeLimits(420, 320, 1200, 1600);
    setSize(560, 760);

    headerLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    headerLabel.setColour(juce::Label::textColourId, text);
    addAndMakeVisible(headerLabel);

    statusLabel.setFont(juce::FontOptions(11.0f));
    statusLabel.setColour(juce::Label::textColourId, textDim);
    addAndMakeVisible(statusLabel);

    for (auto* b : { &loadButton, &resetButton, &modeButton, &passButton }) {
        b->setColour(juce::TextButton::buttonColourId, surface);
        b->setColour(juce::TextButton::textColourOffId, text);
        addAndMakeVisible(*b);
    }
    loadButton.onClick = [this] { chooseTargetsFile(); };
    resetButton.onClick = [this] { proc.engine.resetSinceReset(); };
    passButton.onClick = [this] { proc.engine.resetPass(); };
    modeButton.onClick = [this] {
        holdMode = !holdMode;
        modeButton.setButtonText(holdMode ? "Mode: Hold loudest" : "Mode: Live");
        passButton.setVisible(holdMode);
    };

    viewport.setViewedComponent(&rowContainer, false);
    viewport.setScrollBarsShown(true, false);
    addAndMakeVisible(viewport);

    rebuildRows();
    startTimerHz(10);
}

void SonicMeterEditor::DomainHeader::paint(juce::Graphics& g)
{
    g.setColour(accentLight);
    g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
    g.drawText(title.toUpperCase(), getLocalBounds().withTrimmedTop(8),
               juce::Justification::centredLeft);
}

void SonicMeterEditor::rebuildRows()
{
    rows.clear();
    rowFeatures.clear();
    domainHeaders.clear();
    rowContainer.removeAllChildren();

    const auto& t = proc.targets;
    if (!t.isLoaded())
        return;

    // Group by domain, keeping backend order within each; consensus rows
    // (the "Adjustments") float to the top of their domain.
    std::vector<juce::String> domainOrder;
    for (const auto& r : t.ranges)
        if (std::find(domainOrder.begin(), domainOrder.end(), r.domain) == domainOrder.end())
            domainOrder.push_back(r.domain);

    int y = 0;
    for (const auto& domain : domainOrder) {
        auto header = std::make_unique<DomainHeader>();
        header->title = domain;
        header->setBounds(0, y, 10, 28); // width fixed in resized()
        rowContainer.addAndMakeVisible(*header);
        domainHeaders.push_back(std::move(header));
        y += 28;

        for (const int pass : { 0, 1 }) { // consensus first, then the rest
            for (const auto& r : t.ranges) {
                if (r.domain != domain || (pass == 0) != r.hasConsensus)
                    continue;
                auto row = std::make_unique<sonic::MeterRowComponent>();
                row->setRange(r, sonic::isLiveFeature(r.feature));
                row->setBounds(0, y, 10, sonic::MeterRowComponent::rowHeight);
                rowContainer.addAndMakeVisible(*row);
                rowFeatures.push_back(r.feature);
                rows.push_back(std::move(row));
                y += sonic::MeterRowComponent::rowHeight + 6;
            }
        }
        y += 8;
    }
    rowContainer.setSize(10, y + 8);
    resized();
}

void SonicMeterEditor::timerCallback()
{
    if (seenTargetsVersion != proc.targetsVersion.load()) {
        seenTargetsVersion = proc.targetsVersion.load();
        rebuildRows();
    }

    const auto& t = proc.targets;
    headerLabel.setText(t.isLoaded()
        ? (t.artistName.isNotEmpty() ? t.artistName + " — " + t.trackName : t.trackName)
        : "SonicMeter", juce::dontSendNotification);

    const auto snap = proc.engine.getSnapshot();

    // Hold mode reads the loudest window captured this pass; Live reads the
    // rolling window. Before the first capture, hold mode falls back to live.
    const bool useBest = holdMode && snap.hasBest;

    juce::String status;
    if (!t.isLoaded()) {
        status = "Drop a .sonictargets.json here (exported from SonicConverter) or click Load targets";
    } else {
        if (holdMode)
            status << (snap.hasBest ? "Holding loudest 4s of this pass — play the song through, then adjust and hit New pass.   "
                                    : "Play the song (or the chorus) — capturing the loudest 4s...   ");
        status << "LUFS since reset: "
               << (std::isfinite(snap.sinceResetLufs)
                       ? juce::String(snap.sinceResetLufs, 1) : juce::String("--"))
               << "   True peak since reset: "
               << (snap.sinceResetTruePeakDb > -199.0
                       ? juce::String(snap.sinceResetTruePeakDb, 1) + " dBTP" : juce::String("--"));
        if (!snap.windowFull)
            status << "   (filling window...)";
        else if (!snap.receivingAudio && !useBest)
            status << "   (no audio)";
    }
    statusLabel.setText(status, juce::dontSendNotification);

    if (!t.isLoaded() || (!snap.windowFull && !useBest))
        return;

    const auto values = (useBest ? snap.best : snap.window).asMap();
    const bool stale = !useBest && !snap.receivingAudio;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto it = values.find(rowFeatures[i].toStdString());
        if (it != values.end())
            rows[i]->setValue(it->second, stale);
    }
}

void SonicMeterEditor::paint(juce::Graphics& g)
{
    g.fillAll(bg);
    if (dragHover) {
        g.setColour(accent.withAlpha(0.6f));
        g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(4.0f), 8.0f, 2.0f);
    }
    if (!proc.targets.isLoaded()) {
        g.setColour(textDim);
        g.setFont(juce::FontOptions(14.0f));
        g.drawText("No targets loaded", viewport.getBounds(), juce::Justification::centred);
    }
}

void SonicMeterEditor::resized()
{
    auto area = getLocalBounds().reduced(16);
    auto top = area.removeFromTop(30);
    resetButton.setBounds(top.removeFromRight(90));
    top.removeFromRight(8);
    loadButton.setBounds(top.removeFromRight(110));
    headerLabel.setBounds(top);
    area.removeFromTop(4);
    auto row2 = area.removeFromTop(26);
    passButton.setBounds(row2.removeFromRight(84).reduced(0, 1));
    row2.removeFromRight(8);
    modeButton.setBounds(row2.removeFromRight(140).reduced(0, 1));
    row2.removeFromRight(8);
    statusLabel.setBounds(row2);
    area.removeFromTop(6);
    viewport.setBounds(area);

    const int w = viewport.getMaximumVisibleWidth();
    rowContainer.setSize(w, rowContainer.getHeight());
    for (auto& r : rows)
        r->setSize(w, sonic::MeterRowComponent::rowHeight);
    for (auto& h : domainHeaders)
        h->setSize(w, 28);
}

bool SonicMeterEditor::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (const auto& f : files)
        if (f.endsWithIgnoreCase(".json"))
            return true;
    return false;
}

void SonicMeterEditor::filesDropped(const juce::StringArray& files, int, int)
{
    dragHover = false;
    for (const auto& f : files)
        if (f.endsWithIgnoreCase(".json")) {
            loadTargetsFromFile(juce::File(f));
            break;
        }
    repaint();
}

void SonicMeterEditor::loadTargetsFromFile(const juce::File& f)
{
    if (proc.targets.loadFromJson(f.loadFileAsString())) {
        proc.targetsVersion.fetch_add(1);
        proc.updateHostDisplay();
    } else {
        statusLabel.setText("Could not parse " + f.getFileName(), juce::dontSendNotification);
    }
}

void SonicMeterEditor::chooseTargetsFile()
{
    chooser = std::make_unique<juce::FileChooser>(
        "Load SonicConverter targets", juce::File(), "*.json");
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& fc) {
            if (fc.getResult().existsAsFile())
                loadTargetsFromFile(fc.getResult());
        });
}
