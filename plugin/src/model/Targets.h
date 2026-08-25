#pragma once
#include <juce_core/juce_core.h>
#include <vector>
#include <map>

// .sonictargets.json — exported from the SonicConverter web UI ("Export
// targets for plugin"). Shape mirrors app.py _generate_full_target_ranges:
// one range per production feature, percentile band always present, the
// consensus fields (action/direction/agree) nullable.

namespace sonic {

struct TargetRange {
    juce::String feature, domain, action, unitKind, direction;
    double you = 0.0;
    double targetCohort = 0.0;
    bool hasSignature = false;
    double targetSignature = 0.0;
    bool hasPercentiles = false;
    double p5 = 0, p25 = 0, p50 = 0, p75 = 0, p95 = 0;
    bool hasConsensus = false;
    int agreeCount = 0, agreeTotal = 0;
};

struct TargetsFile {
    juce::String trackName, artistName, jobId, exportedAt;
    std::map<juce::String, double> staticFeatures; // bpm etc. (numeric only)
    juce::String key, scale;                       // non-numeric statics
    std::vector<TargetRange> ranges;
    juce::String rawJson; // persisted verbatim in plugin state

    bool loadFromJson(const juce::String& jsonText);
    bool isLoaded() const { return !ranges.empty(); }
};

// recBand() port: resolve the band (synthetic fallback when no percentiles)
// and whether a value sits in the p25–p75 zone (with the negligible-move fold).
struct Band {
    double p5, p25, p50, p75, p95;
    bool inZone;
};
Band resolveBand(const TargetRange& r, double liveValue);

// fmtFeatVal / fmtMove / fmtRange ports (unit-aware display strings)
juce::String fmtFeatVal(const juce::String& kind, double v);
juce::String fmtMove(const juce::String& kind, double you, double target);
juce::String fmtRange(const juce::String& kind, double a, double b);

// Short human label for a feature key ("bass_ratio" -> "Bass 60-250 Hz")
juce::String featureLabel(const juce::String& feature);

// Features the engine computes live; everything else renders as a static row.
bool isLiveFeature(const juce::String& feature);

} // namespace sonic
