#include "Targets.h"
#include <cmath>

namespace sonic {

bool TargetsFile::loadFromJson(const juce::String& jsonText)
{
    const juce::var root = juce::JSON::parse(jsonText);
    if (!root.isObject())
        return false;

    const juce::var rangesVar = root["ranges"];
    if (!rangesVar.isArray() || rangesVar.size() == 0)
        return false;

    ranges.clear();
    staticFeatures.clear();

    const juce::var track = root["track"];
    trackName = track["name"].toString();
    artistName = track["artist"].toString();
    jobId = track["job_id"].toString();
    exportedAt = root["exported_at"].toString();

    if (const auto* sf = root["static_features"].getDynamicObject()) {
        for (const auto& prop : sf->getProperties()) {
            const juce::String name = prop.name.toString();
            if (prop.value.isDouble() || prop.value.isInt() || prop.value.isInt64())
                staticFeatures[name] = (double) prop.value;
            else if (name == "key")
                key = prop.value.toString();
            else if (name == "scale")
                scale = prop.value.toString();
        }
    }

    for (int i = 0; i < rangesVar.size(); ++i) {
        const juce::var rv = rangesVar[i];
        if (!rv.isObject())
            continue;
        TargetRange r;
        r.feature = rv["feature"].toString();
        r.domain = rv["domain"].toString();
        r.unitKind = rv["unit_kind"].toString();
        r.you = (double) rv["you"];
        r.targetCohort = (double) rv["target_cohort"];
        if (!rv["target_signature"].isVoid()) {
            r.hasSignature = true;
            r.targetSignature = (double) rv["target_signature"];
        }
        if (rv["percentiles"].isObject()) {
            const juce::var p = rv["percentiles"];
            r.hasPercentiles = true;
            r.p5 = (double) p["p5"];
            r.p25 = (double) p["p25"];
            r.p50 = (double) p["p50"];
            r.p75 = (double) p["p75"];
            r.p95 = (double) p["p95"];
        }
        if (!rv["direction"].isVoid() && rv["direction"].toString().isNotEmpty()) {
            r.hasConsensus = true;
            r.direction = rv["direction"].toString();
            r.action = rv["action"].toString();
            const juce::var agree = rv["agree"];
            if (agree.isArray() && agree.size() == 2) {
                r.agreeCount = (int) agree[0];
                r.agreeTotal = (int) agree[1];
            }
        }
        ranges.push_back(std::move(r));
    }

    rawJson = jsonText;
    return isLoaded();
}

// --- recMoveNegligible port (app.js) ---
static bool moveNegligible(const juce::String& kind, double you, double edge)
{
    if (std::isnan(you) || std::isnan(edge))
        return false;
    if (kind == "pct")
        return (you > 0 && edge > 0) ? std::abs(10.0 * std::log10(edge / you)) < 0.05
                                     : std::abs((edge - you) * 100.0) < 0.05;
    if (kind == "db" || kind == "lufs" || kind == "lu" || kind == "rate" || kind == "ms")
        return std::abs(edge - you) < 0.05;
    if (kind == "hz")
        return std::abs(edge - you) < 0.5;
    return you != 0.0 && std::abs((edge - you) / std::abs(you)) * 100.0 < 1.5;
}

Band resolveBand(const TargetRange& r, double you)
{
    Band b {};
    if (r.hasPercentiles) {
        b.p5 = r.p5; b.p25 = r.p25; b.p50 = r.p50; b.p75 = r.p75; b.p95 = r.p95;
    } else {
        const double t = r.targetCohort;
        double sp = std::abs(t - you);
        if (sp == 0.0) sp = std::abs(t) * 0.1 + 1e-6;
        b.p25 = t - sp * 0.15; b.p75 = t + sp * 0.15; b.p50 = t;
        b.p5 = t - sp * 0.6; b.p95 = t + sp * 0.6;
    }
    b.inZone = you >= b.p25 && you <= b.p75;
    if (!b.inZone) {
        const double edge = you < b.p25 ? b.p25 : b.p75;
        if (moveNegligible(r.unitKind, you, edge))
            b.inZone = true;
    }
    return b;
}

// --- display formatting ports ---
juce::String fmtFeatVal(const juce::String& kind, double v)
{
    if (std::isnan(v))
        return juce::String::fromUTF8("–");
    auto f = [](double x, int dp) { return juce::String(x, dp); };
    if (kind == "pct")  return f(v * 100.0, 1) + "%";
    if (kind == "db")   return f(v, 1) + " dB";
    if (kind == "lufs") return f(v, 1) + " LUFS";
    if (kind == "hz")   return juce::String((int) std::round(v)) + " Hz";
    if (kind == "rate") return f(v, 1) + " /s";
    if (kind == "ms")   return f(v, 1) + " ms";
    if (kind == "lu")   return f(v, 1) + " LU";
    return f(v, 3);
}

static juce::String signedStr(double v, int dp, const juce::String& unit)
{
    const juce::String sign = v >= 0 ? "+" : juce::String::fromUTF8("−");
    return sign + juce::String(std::abs(v), dp) + unit;
}

juce::String fmtMove(const juce::String& kind, double you, double target)
{
    if (std::isnan(you) || std::isnan(target))
        return {};
    if (kind == "pct") {
        if (you > 0 && target > 0)
            return signedStr(10.0 * std::log10(target / you), 1, " dB");
        return signedStr((target - you) * 100.0, 1, " pts");
    }
    if (kind == "db" || kind == "lufs") return signedStr(target - you, 1, " dB");
    if (kind == "lu")   return signedStr(target - you, 1, " LU");
    if (kind == "hz")
        return (target >= you ? juce::String("+") : juce::String::fromUTF8("−"))
               + juce::String((int) std::round(std::abs(target - you))) + " Hz";
    if (kind == "rate") return signedStr(target - you, 1, " /s");
    if (kind == "ms")   return signedStr(target - you, 1, " ms");
    return you != 0.0 ? signedStr(((target - you) / std::abs(you)) * 100.0, 0, "%") : juce::String();
}

juce::String fmtRange(const juce::String& kind, double a, double b)
{
    const juce::String A = fmtFeatVal(kind, a), B = fmtFeatVal(kind, b);
    // strip the unit off the low end when both match: "17.1–19.0%"
    int split = 0;
    while (split < A.length()
           && (juce::CharacterFunctions::isDigit(A[split]) || A[split] == '.'
               || A[split] == '-' || A[split] == 0x2212))
        ++split;
    const juce::String unit = A.substring(split);
    if (unit.isNotEmpty() && B.endsWith(unit))
        return A.substring(0, split) + juce::String::fromUTF8("–") + B;
    return A + juce::String::fromUTF8("–") + B;
}

juce::String featureLabel(const juce::String& f)
{
    static const std::map<juce::String, juce::String> labels = {
        { "sub_ratio", "Sub 20-60 Hz" },
        { "bass_ratio", "Bass 60-250 Hz" },
        { "low_mid_ratio", "Low mids 250-500 Hz" },
        { "mid_ratio", "Mids 500 Hz-2 kHz" },
        { "high_mid_ratio", "High mids 2-4 kHz" },
        { "presence_ratio", "Presence 4-8 kHz" },
        { "air_ratio", "Air 8-20 kHz" },
        { "brightness", "Brightness (centroid)" },
        { "spectral_rolloff", "Spectral rolloff" },
        { "brightness_variance", "Brightness variance" },
        { "energy", "RMS energy" },
        { "beat_strength", "Beat strength" },
        { "onset_rate", "Onset rate" },
        { "attack_time", "Attack time" },
        { "danceability", "Danceability" },
        { "lufs_integrated", "Loudness (LUFS)" },
        { "dynamic_range", "Dynamic range" },
        { "loudness_range", "Loudness range" },
        { "crest_factor", "Crest factor" },
        { "compression_amount", "Compression" },
        { "spectral_complexity", "Spectral complexity" },
        { "dissonance", "Dissonance" },
        { "key_strength", "Key strength" },
        { "zcr", "Zero-crossing rate" },
        { "spectral_flux", "Spectral flux" },
        { "harmonic_distortion", "Harmonic distortion" },
        { "stereo_width", "Stereo width" },
        { "mid_side_ratio", "Mid/side ratio" },
        { "stereo_correlation", "Stereo correlation" },
        { "true_peak_dbfs", "True peak" },
    };
    const auto it = labels.find(f);
    return it != labels.end() ? it->second : f;
}

bool isLiveFeature(const juce::String& f)
{
    static const std::map<juce::String, bool> statics = {
        { "beat_strength", true }, { "onset_rate", true }, { "attack_time", true },
        { "danceability", true }, { "key_strength", true },
    };
    return statics.find(f) == statics.end();
}

} // namespace sonic
