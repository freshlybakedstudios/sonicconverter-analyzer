#include "MeterRowComponent.h"
#include <cmath>

namespace sonic {

static int stringWidth(const juce::Font& f, const juce::String& s)
{
    juce::GlyphArrangement ga;
    ga.addLineOfText(f, s, 0.0f, 0.0f);
    return juce::roundToInt(ga.getBoundingBox(0, -1, true).getWidth());
}

void MeterRowComponent::setRange(const TargetRange& r, bool live)
{
    range = r;
    isLive = live;
    if (!live) {
        value = r.you; // static rows show the analysis-time value
        hasValue = true;
        stale = true;
    }
    repaint();
}

void MeterRowComponent::setValue(double v, bool isStale)
{
    if (!isLive)
        return;
    if (hasValue && std::abs(v - value) < 1e-9 && isStale == stale)
        return;
    value = v;
    hasValue = true;
    stale = isStale;
    repaint();
}

void MeterRowComponent::paint(juce::Graphics& g)
{
    const auto area = getLocalBounds().reduced(0, 4);
    const double you = hasValue ? value : range.you;
    const Band band = resolveBand(range, you);

    // --- scale math, straight from recRangeRow ---
    double span = band.p95 - band.p5;
    if (span == 0.0)
        span = std::abs(band.p95) * 0.1 + 1e-6;
    const double pad = span * 0.06;
    const double sMin = band.p5 - pad, sMax = band.p95 + pad;
    auto pos = [&](double v) {
        const double raw = (v - sMin) / (sMax - sMin) * 100.0;
        return juce::jlimit(0.0, 100.0, raw);
    };

    // --- header line ---
    auto header = area.withHeight(18);
    g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
    const juce::String domain = range.domain.toUpperCase();
    const int chipW = stringWidth(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)), domain) + 14;

    auto chip = header.removeFromLeft(juce::jmax(chipW, 30)).toFloat();
    g.setColour(theme::accent.withAlpha(0.25f));
    g.fillRoundedRectangle(chip.reduced(0, 2), 4.0f);
    g.setColour(theme::accentLight);
    g.drawText(domain, chip.toNearestInt(), juce::Justification::centred);

    header.removeFromLeft(6);

    // move headline (right-aligned)
    const double edge = you < band.p25 ? band.p25 : band.p75;
    const bool inZone = band.inZone;
    const juce::String moveStr = !hasValue ? juce::String("waiting for audio")
        : inZone ? juce::String::fromUTF8("\xE2\x9C\x93 in the zone")
                 : fmtMove(range.unitKind, you, edge) + " to land in";
    g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
    g.setColour(!hasValue ? theme::textDim : inZone ? theme::success : theme::accent);
    const int moveW = stringWidth(juce::Font(juce::FontOptions(12.0f, juce::Font::bold)), moveStr) + 4;
    g.drawText(moveStr, header.removeFromRight(moveW), juce::Justification::centredRight);

    // label (+ static badge)
    g.setFont(juce::FontOptions(13.0f));
    g.setColour(theme::text);
    juce::String label = featureLabel(range.feature);
    if (!isLive)
        label += "  (from analysis)";
    else if (stale && hasValue)
        label += "  (held)";
    g.drawText(label, header, juce::Justification::centredLeft, true);

    // --- bar ---
    auto barArea = area.withTrimmedTop(22).withHeight(14).toFloat();
    g.setColour(theme::border);
    g.fillRoundedRectangle(barArea, 7.0f);

    // striped target zone p25-p75
    const float zoneL = (float) pos(band.p25) / 100.0f * barArea.getWidth();
    float zoneW = (float) pos(band.p75) / 100.0f * barArea.getWidth() - zoneL;
    zoneW = juce::jmax(zoneW, barArea.getWidth() * 0.015f);
    {
        juce::Graphics::ScopedSaveState ss(g);
        const juce::Rectangle<float> zone(barArea.getX() + zoneL, barArea.getY(), zoneW, barArea.getHeight());
        juce::Path clip;
        clip.addRoundedRectangle(zone, 3.0f);
        g.reduceClipRegion(clip);
        g.setColour(theme::accent.withAlpha(0.5f));
        for (float x = zone.getX() - barArea.getHeight(); x < zone.getRight(); x += 7.0f)
            g.drawLine(x, zone.getBottom() + 2.0f, x + zone.getHeight() + 4.0f, zone.getY() - 2.0f, 3.0f);
    }

    // median tick
    g.setColour(theme::accentLight);
    const float mx = barArea.getX() + (float) pos(band.p50) / 100.0f * barArea.getWidth();
    g.fillRect(juce::Rectangle<float>(mx - 1.0f, barArea.getY(), 2.0f, barArea.getHeight()));

    // signature target tick (the "distinctive winners" edge)
    if (range.hasSignature) {
        g.setColour(theme::text.withAlpha(0.55f));
        const float sx = barArea.getX() + (float) pos(range.targetSignature) / 100.0f * barArea.getWidth();
        g.fillRect(juce::Rectangle<float>(sx - 1.0f, barArea.getY(), 2.0f, barArea.getHeight()));
    }

    // live dot (clamped 2..98%, off-scale keeps the clamp like the web UI)
    if (hasValue) {
        const double dotPct = juce::jlimit(2.0, 98.0, pos(you));
        const float dx = barArea.getX() + (float) dotPct / 100.0f * barArea.getWidth();
        const float dy = barArea.getCentreY();
        g.setColour(theme::card);
        g.fillEllipse(dx - 9.0f, dy - 9.0f, 18.0f, 18.0f);
        g.setColour(stale ? theme::accentLight.withAlpha(0.6f) : theme::accentLight);
        g.fillEllipse(dx - 7.0f, dy - 7.0f, 14.0f, 14.0f);
    }

    // --- legend ---
    auto legend = area.withTrimmedTop(40).withHeight(16);
    g.setFont(juce::FontOptions(11.0f));
    auto drawLegend = [&](const juce::String& dim, const juce::String& val, juce::Colour valCol) {
        g.setColour(theme::textDim);
        const juce::Font dimF(juce::FontOptions(11.0f));
        const int w1 = stringWidth(dimF, dim);
        g.drawText(dim, legend.removeFromLeft(w1 + 2), juce::Justification::centredLeft);
        g.setColour(valCol);
        const juce::Font valF(juce::FontOptions(11.0f, juce::Font::bold));
        g.setFont(valF);
        const int w2 = stringWidth(valF, val);
        g.drawText(val, legend.removeFromLeft(w2 + 2), juce::Justification::centredLeft);
        g.setFont(dimF);
        legend.removeFromLeft(12);
    };
    drawLegend(isLive ? "Live " : "You ", hasValue ? fmtFeatVal(range.unitKind, you) : "--",
               theme::accentLight);
    drawLegend("Target zone ", fmtRange(range.unitKind, band.p25, band.p75), theme::accent);
    if (range.hasSignature)
        drawLegend("Signature ", fmtFeatVal(range.unitKind, range.targetSignature),
                   theme::text.withAlpha(0.75f));
    if (range.hasConsensus)
        drawLegend("", juce::String(range.agreeCount) + "/" + juce::String(range.agreeTotal)
                       + " agree", theme::textDim);
}

} // namespace sonic
