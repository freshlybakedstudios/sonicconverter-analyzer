cmake_minimum_required(VERSION 3.24)
project(SonicMeter VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")

include(FetchContent)

FetchContent_Declare(
  JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.14
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(JUCE)

# kissfft: we only need the raw sources — compile them ourselves (double
# precision for numpy parity) instead of using its CMake build.
FetchContent_Declare(
  kissfft_src
  GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
  GIT_TAG v131
  GIT_SHALLOW ON
)
FetchContent_Populate(kissfft_src)

add_library(kissfft_double STATIC
  ${kissfft_src_SOURCE_DIR}/kiss_fft.c
  ${kissfft_src_SOURCE_DIR}/tools/kiss_fftr.c
)
target_include_directories(kissfft_double PUBLIC
  ${kissfft_src_SOURCE_DIR}
  ${kissfft_src_SOURCE_DIR}/tools
)
target_compile_definitions(kissfft_double PUBLIC kiss_fft_scalar=double)

# --- Feature engine (UI-free, shared by plugin and parity_cli) ---
add_library(sonic_engine STATIC
  src/engine/FeatureMath.cpp
  src/engine/Loudness.cpp
  src/engine/StftFeatures.cpp
  src/engine/BandAnalyzer.cpp
)
target_include_directories(sonic_engine PUBLIC src)
target_link_libraries(sonic_engine PUBLIC kissfft_double)
target_compile_features(sonic_engine PUBLIC cxx_std_17)

# --- The plugin ---
juce_add_plugin(SonicMeter
  COMPANY_NAME "Freshly Baked Studios"
  BUNDLE_ID com.freshlybakedstudios.sonicmeter
  PLUGIN_MANUFACTURER_CODE Fbks
  PLUGIN_CODE Sonm
  FORMATS AU VST3 Standalone
  PRODUCT_NAME "SonicMeter"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  COPY_PLUGIN_AFTER_BUILD TRUE
  VST3_CATEGORIES "Analyzer"
)

target_sources(SonicMeter PRIVATE
  src/PluginProcessor.cpp
  src/PluginEditor.cpp
  src/engine/FeatureEngine.cpp
  src/model/Targets.cpp
  src/ui/MeterRowComponent.cpp
)

target_compile_definitions(SonicMeter PUBLIC
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(SonicMeter
  PRIVATE
    sonic_engine
    juce::juce_audio_utils
    juce::juce_dsp
  PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# --- Parity CLI: WAV in, features JSON out (same engine code) ---
juce_add_console_app(parity_cli PRODUCT_NAME "parity_cli")
target_sources(parity_cli PRIVATE tools/parity_cli.cpp)
target_link_libraries(parity_cli PRIVATE
  sonic_engine
  juce::juce_audio_formats
  juce::juce_recommended_config_flags
)
