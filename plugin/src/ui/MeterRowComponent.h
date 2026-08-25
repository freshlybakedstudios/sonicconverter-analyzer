#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "../model/Targets.h"

// One feature row: header (domain chip, label/action, move headline), the
// range bar (striped p25-p75 zone, p50 median tick, live dot), and a legend.
// A native port of the web UI's recRangeRow markup + .rec-range* styles.

namespace sonic {

namespace theme {
    inline const juce::Colour bg          { 0xff141213 };
    inline const juce::Colour surface     { 0xff1a1818 };
    inline const juce::Colour card        { 0xff231f20 };
    inline const juce::Colour border      { 0xff3a3636 };
    inline const juce::Colour text        { 0xffe8e8f0 };
    inline const juce::Colour textDim     { 0xff888899 };
    inline const juce::Colour accent      { 0xffD8E166 };
    inline const juce::Colour accentLight { 0xffB5C851 };
    inline const juce::Colour success     { 0xffB0C936 };
    inline const juce::Colour danger      { 0xffff6b6b };
}

class MeterRowComponent : public juce::Component {
public:
    static constexpr int rowHeight = 64;

    void setRange(const TargetRange& r, bool live);
    // Live value from the engine (or the analysis-time value for static rows)
    void setValue(double v, bool isStale);

    void paint(juce::Graphics& g) override;

private:
    TargetRange range;
    bool isLive = false;
    double value = 0.0;
    bool stale = true;
    bool hasValue = false;
};

} // namespace sonic
