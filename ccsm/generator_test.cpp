#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <iostream>
#include <memory>

// Basic test for generator functionality
// Compile with: g++ generator_test.cpp -o generator_test -I/path/to/ccsm/include -L/path/to/ccsm/lib -lccsm_core -std=c++17

int main() {
    // Load CSM-1B model
    std::shared_ptr<ccsm::Generator> generator = ccsm::load_csm_1b("cpu");
    
    // Basic generation test
    std::string text = "Hello, world! This is a test of text to speech.";
    std::vector<float> audio = generator->generate_speech(text, 0);
    
    // Print basic stats
    std::cout << "Generated " << audio.size() << " audio samples" << std::endl;
    std::cout << "Sample rate: " << generator->sample_rate() << " Hz" << std::endl;
    
    // Test with different temperature
    std::vector<float> audio2 = generator->generate_speech(text, 0, 0.5f, 30);
    
    // Print comparison
    std::cout << "Second generation: " << audio2.size() << " audio samples" << std::endl;
    
    // Test with context
    std::vector<ccsm::Segment> context;
    context.push_back(ccsm::Segment("This is some context.", 0));
    std::vector<float> audio3 = generator->generate_speech("Continuing the conversation.", 0, context);
    
    std::cout << "Generation with context: " << audio3.size() << " audio samples" << std::endl;
    
    return 0;
}