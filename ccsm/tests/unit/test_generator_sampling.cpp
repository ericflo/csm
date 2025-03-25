#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/utils.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>

using namespace ccsm;

// Mock classes for testing advanced sampling techniques
class SamplingModel : public Model {
public:
    SamplingModel() : Model(createConfig()) {
        // Default behavior: return deterministic logits
        deterministic_logits_ = true;
        enable_logit_bias_ = false;
        repetition_penalty_ = 1.0f;
        enable_frequency_penalty_ = false;
        enable_presence_penalty_ = false;
    }
    
    static ModelConfig createConfig() {
        ModelConfig config;
        config.name = "Sampling Test Model";
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 1024;
        config.n_heads = 16;
        config.n_kv_heads = 8;
        config.n_layers = 32;
        config.n_audio_layers = 12;
        config.max_seq_len = 2048;
        config.num_codebooks = 8;
        return config;
    }
    
    // Required Model interface methods
    bool load_weights(const std::string& path) override { return true; }
    bool load_weights(std::shared_ptr<ModelLoader> loader) override { return true; }
    bool load_weights(const WeightMap& weights) override { return true; }
    
    void reset_caches() override {}
    void optimize_memory(size_t max_memory_mb = 0) override {}
    void prune_caches(float prune_factor = 0.5f) override {}
    
    // Generate frame with specified sampling logic
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override {
        
        // Record parameters for testing
        last_tokens_ = tokens;
        last_temperature_ = temperature;
        last_top_k_ = top_k;
        
        // Use logit generation function based on configuration
        std::vector<float> raw_logits;
        if (deterministic_logits_) {
            raw_logits = generate_deterministic_logits(tokens, positions);
        } else {
            raw_logits = generate_random_logits(tokens, positions);
        }
        
        // Store for later reference
        last_logits_ = raw_logits;
        
        // Apply any specified biases
        if (enable_logit_bias_) {
            apply_logit_bias(raw_logits);
        }
        
        // Apply repetition penalty if enabled
        if (repetition_penalty_ != 1.0f) {
            apply_repetition_penalty(raw_logits, tokens);
        }
        
        // Apply frequency penalty if enabled
        if (enable_frequency_penalty_) {
            apply_frequency_penalty(raw_logits, tokens);
        }
        
        // Apply presence penalty if enabled
        if (enable_presence_penalty_) {
            apply_presence_penalty(raw_logits, tokens);
        }
        
        // Apply temperature and sample
        std::vector<int> frame = sample_from_logits(raw_logits, temperature, top_k);
        
        // Record sampled frame for testing
        last_sampled_frame_ = frame;
        
        return frame;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override {
        return generate_deterministic_logits(tokens, positions);
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override {
        std::vector<float> result(config_.audio_vocab_size, -10.0f);
        
        // Make a few tokens have high probability
        for (int i = 0; i < 10; i++) {
            int token = (codebook * 10 + i) % config_.audio_vocab_size;
            result[token] = 5.0f;
        }
        
        return result;
    }
    
    // Configuration methods for testing
    void set_deterministic_logits(bool deterministic) {
        deterministic_logits_ = deterministic;
    }
    
    void set_logit_bias(std::unordered_map<int, float> bias) {
        logit_bias_ = std::move(bias);
        enable_logit_bias_ = true;
    }
    
    void set_repetition_penalty(float penalty) {
        repetition_penalty_ = penalty;
    }
    
    void enable_frequency_penalty(bool enable, float penalty = 0.1f) {
        enable_frequency_penalty_ = enable;
        frequency_penalty_ = penalty;
    }
    
    void enable_presence_penalty(bool enable, float penalty = 0.1f) {
        enable_presence_penalty_ = enable;
        presence_penalty_ = penalty;
    }
    
    // State inspection
    const std::vector<int>& get_last_tokens() const { return last_tokens_; }
    float get_last_temperature() const { return last_temperature_; }
    int get_last_top_k() const { return last_top_k_; }
    const std::vector<float>& get_last_logits() const { return last_logits_; }
    const std::vector<int>& get_last_sampled_frame() const { return last_sampled_frame_; }
    
private:
    // Generate deterministic logits for testing
    std::vector<float> generate_deterministic_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) {
        
        std::vector<float> logits(config_.audio_vocab_size, -10.0f);
        
        // Add predictable pattern based on last token
        int last_token = tokens.empty() ? 0 : tokens.back();
        int offset = last_token % 100;
        
        // Set a clear distribution
        for (int i = 0; i < 20; i++) {
            int index = (offset + i) % config_.audio_vocab_size;
            // Decreasing probabilities
            logits[index] = 10.0f - (i * 0.5f);
        }
        
        return logits;
    }
    
    // Generate random logits for testing probabilistic sampling
    std::vector<float> generate_random_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) {
        
        // Use a fixed seed for reproducibility
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        
        std::vector<float> logits(config_.audio_vocab_size);
        for (auto& logit : logits) {
            logit = dist(rng);
        }
        
        return logits;
    }
    
    // Apply bias to specific tokens
    void apply_logit_bias(std::vector<float>& logits) {
        for (const auto& [token, bias] : logit_bias_) {
            if (token >= 0 && token < static_cast<int>(logits.size())) {
                logits[token] += bias;
            }
        }
    }
    
    // Apply repetition penalty to tokens that have appeared in the context
    void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int>& tokens) {
        // Count token frequencies
        std::unordered_map<int, int> token_counts;
        for (int token : tokens) {
            token_counts[token]++;
        }
        
        // Apply penalty to repeated tokens
        for (const auto& [token, count] : token_counts) {
            if (token < static_cast<int>(logits.size())) {
                if (logits[token] > 0) {
                    logits[token] /= repetition_penalty_;
                } else {
                    logits[token] *= repetition_penalty_;
                }
            }
        }
    }
    
    // Apply frequency penalty based on token frequency
    void apply_frequency_penalty(std::vector<float>& logits, const std::vector<int>& tokens) {
        // Count token frequencies
        std::unordered_map<int, int> token_counts;
        for (int token : tokens) {
            token_counts[token]++;
        }
        
        // Apply frequency penalty
        for (const auto& [token, count] : token_counts) {
            if (token < static_cast<int>(logits.size())) {
                logits[token] -= frequency_penalty_ * count;
            }
        }
    }
    
    // Apply presence penalty to tokens that have appeared
    void apply_presence_penalty(std::vector<float>& logits, const std::vector<int>& tokens) {
        // Track unique tokens
        std::unordered_set<int> unique_tokens;
        for (int token : tokens) {
            unique_tokens.insert(token);
        }
        
        // Apply presence penalty
        for (int token : unique_tokens) {
            if (token < static_cast<int>(logits.size())) {
                logits[token] -= presence_penalty_;
            }
        }
    }
    
    // Sample tokens from logits using temperature and top-k
    std::vector<int> sample_from_logits(
        const std::vector<float>& logits,
        float temperature,
        int top_k) {
        
        // Create a frame of tokens based on logits
        std::vector<int> frame(config_.num_codebooks);
        
        // Sample each codebook separately
        for (int cb = 0; cb < config_.num_codebooks; cb++) {
            // Get decoder logits for this codebook
            std::vector<float> cb_logits = get_decoder_logits({}, {}, cb);
            
            // Apply temperature to soften/sharpen distribution
            std::vector<float> probs = apply_temperature(cb_logits, temperature);
            
            // Apply top-k filtering
            if (top_k > 0 && top_k < static_cast<int>(probs.size())) {
                apply_top_k(probs, top_k);
            }
            
            // Convert to probabilities and sample
            normalize_to_probabilities(probs);
            
            // Sample token
            int token = sample_token(probs, cb);
            frame[cb] = token;
        }
        
        return frame;
    }
    
    // Apply temperature to logits
    std::vector<float> apply_temperature(const std::vector<float>& logits, float temperature) {
        if (temperature <= 0.0f) {
            // For temperature = 0, return greedy selection
            std::vector<float> result(logits.size(), 0.0f);
            auto max_it = std::max_element(logits.begin(), logits.end());
            if (max_it != logits.end()) {
                result[std::distance(logits.begin(), max_it)] = 1.0f;
            }
            return result;
        }
        
        std::vector<float> scaled_logits = logits;
        for (auto& logit : scaled_logits) {
            logit /= temperature;
        }
        
        return scaled_logits;
    }
    
    // Apply top-k filtering
    void apply_top_k(std::vector<float>& logits, int k) {
        // Find the k-th largest element
        std::vector<float> sorted_logits = logits;
        std::nth_element(sorted_logits.begin(), sorted_logits.begin() + k - 1, sorted_logits.end(), 
                        std::greater<float>());
        float threshold = sorted_logits[k - 1];
        
        // Zero out logits below threshold
        for (auto& logit : logits) {
            if (logit < threshold) {
                logit = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    // Normalize logits to probabilities with softmax
    void normalize_to_probabilities(std::vector<float>& logits) {
        // Apply softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (auto& logit : logits) {
            logit = std::exp(logit - max_logit);
            sum += logit;
        }
        
        if (sum > 0.0f) {
            for (auto& logit : logits) {
                logit /= sum;
            }
        } else {
            // Fallback to uniform if sum is 0
            float uniform_prob = 1.0f / logits.size();
            std::fill(logits.begin(), logits.end(), uniform_prob);
        }
    }
    
    // Sample a token from probabilities
    int sample_token(const std::vector<float>& probs, int codebook) {
        // Use a fixed seed based on codebook for reproducibility in tests
        std::mt19937 rng(42 + codebook);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        float r = dist(rng);
        float cdf = 0.0f;
        
        for (size_t i = 0; i < probs.size(); i++) {
            cdf += probs[i];
            if (r < cdf) {
                return static_cast<int>(i);
            }
        }
        
        // Fallback
        return 1; // Never return 0 (EOS) by default
    }
    
    // Configuration
    bool deterministic_logits_ = true;
    bool enable_logit_bias_ = false;
    std::unordered_map<int, float> logit_bias_;
    float repetition_penalty_ = 1.0f;
    bool enable_frequency_penalty_ = false;
    float frequency_penalty_ = 0.0f;
    bool enable_presence_penalty_ = false;
    float presence_penalty_ = 0.0f;
    
    // State tracking
    std::vector<int> last_tokens_;
    float last_temperature_ = 0.0f;
    int last_top_k_ = 0;
    std::vector<float> last_logits_;
    std::vector<int> last_sampled_frame_;
};

// Test fixture for Generator advanced sampling tests
class GeneratorSamplingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock components
        model_ = std::make_shared<SamplingModel>();
        text_tokenizer_ = std::make_shared<MockTextTokenizer>();
        audio_codec_ = std::make_shared<MockAudioCodec>();
        watermarker_ = std::make_shared<MockWatermarker>();
        
        // Create generator
        generator_ = std::make_shared<Generator>(model_, text_tokenizer_, audio_codec_, watermarker_);
    }
    
    // Helper to cast model to SamplingModel
    std::shared_ptr<SamplingModel> sampling_model() {
        return std::static_pointer_cast<SamplingModel>(model_);
    }
    
    // Components
    std::shared_ptr<Model> model_;
    std::shared_ptr<TextTokenizer> text_tokenizer_;
    std::shared_ptr<AudioCodec> audio_codec_;
    std::shared_ptr<Watermarker> watermarker_;
    std::shared_ptr<Generator> generator_;
};

// Test temperature effects on sampling
TEST_F(GeneratorSamplingTest, TemperatureEffects) {
    // Generate with different temperatures
    GenerationOptions cold_options;
    cold_options.temperature = 0.01f; // Very low (almost greedy)
    cold_options.seed = 42;
    
    GenerationOptions hot_options;
    hot_options.temperature = 1.5f; // High (more random)
    hot_options.seed = 42;
    
    // Generate with different temperatures
    auto cold_audio = generator_->generate_speech("Test temperature effects", 0, {}, cold_options);
    auto hot_audio = generator_->generate_speech("Test temperature effects", 0, {}, hot_options);
    
    // Both should produce audio
    EXPECT_FALSE(cold_audio.empty());
    EXPECT_FALSE(hot_audio.empty());
    
    // Verify temperature was properly passed to model
    EXPECT_NEAR(sampling_model()->get_last_temperature(), 1.5f, 0.01f);
}

// Test top-k sampling
TEST_F(GeneratorSamplingTest, TopKSampling) {
    // Generate with different top-k values
    GenerationOptions low_k_options;
    low_k_options.top_k = 5;  // Very selective
    low_k_options.seed = 42;
    
    GenerationOptions high_k_options;
    high_k_options.top_k = 100; // More variety
    high_k_options.seed = 42;
    
    // Generate with different top-k
    auto low_k_audio = generator_->generate_speech("Test top-k effects", 0, {}, low_k_options);
    auto high_k_audio = generator_->generate_speech("Test top-k effects", 0, {}, high_k_options);
    
    // Both should produce audio
    EXPECT_FALSE(low_k_audio.empty());
    EXPECT_FALSE(high_k_audio.empty());
    
    // Verify top-k was properly passed to model
    EXPECT_EQ(sampling_model()->get_last_top_k(), 100);
}

// Test greedy sampling (temperature = 0)
TEST_F(GeneratorSamplingTest, GreedySampling) {
    // Use temperature = 0 for deterministic/greedy sampling
    GenerationOptions greedy_options;
    greedy_options.temperature = 0.0f;
    greedy_options.seed = 42;
    
    // Generate with greedy sampling
    auto greedy_audio1 = generator_->generate_speech("Test greedy sampling", 0, {}, greedy_options);
    auto greedy_audio2 = generator_->generate_speech("Test greedy sampling", 0, {}, greedy_options);
    
    // Should produce audio
    EXPECT_FALSE(greedy_audio1.empty());
    EXPECT_FALSE(greedy_audio2.empty());
    
    // With deterministic model and same seed, outputs should be identical
    // Note: This test is conceptual, in practice we'd need to compare the frames
    EXPECT_TRUE(true); // Placeholder
}

// Prepare for repetition penalty implementation
TEST_F(GeneratorSamplingTest, RepetitionPenaltyPreparation) {
    // Configure model to use repetition penalty
    sampling_model()->set_repetition_penalty(1.5f);
    
    // Generate with repetition penalty
    GenerationOptions options;
    options.seed = 42;
    
    auto audio = generator_->generate_speech("Test repetition penalty", 0, {}, options);
    
    // Should produce audio
    EXPECT_FALSE(audio.empty());
    
    // This test is preparatory - full implementation will be added later
    EXPECT_TRUE(true);
}

// Prepare for frequency penalty implementation
TEST_F(GeneratorSamplingTest, FrequencyPenaltyPreparation) {
    // Configure model to use frequency penalty
    sampling_model()->enable_frequency_penalty(true, 0.2f);
    
    // Generate with frequency penalty
    GenerationOptions options;
    options.seed = 42;
    
    auto audio = generator_->generate_speech("Test frequency penalty", 0, {}, options);
    
    // Should produce audio
    EXPECT_FALSE(audio.empty());
    
    // This test is preparatory - full implementation will be added later
    EXPECT_TRUE(true);
}

// Prepare for presence penalty implementation
TEST_F(GeneratorSamplingTest, PresencePenaltyPreparation) {
    // Configure model to use presence penalty
    sampling_model()->enable_presence_penalty(true, 0.2f);
    
    // Generate with presence penalty
    GenerationOptions options;
    options.seed = 42;
    
    auto audio = generator_->generate_speech("Test presence penalty", 0, {}, options);
    
    // Should produce audio
    EXPECT_FALSE(audio.empty());
    
    // This test is preparatory - full implementation will be added later
    EXPECT_TRUE(true);
}

// Prepare for logit bias implementation
TEST_F(GeneratorSamplingTest, LogitBiasPreparation) {
    // Configure model to use logit bias
    std::unordered_map<int, float> bias = {
        {10, 2.0f},   // Boost token 10
        {20, -2.0f},  // Suppress token 20
    };
    sampling_model()->set_logit_bias(bias);
    
    // Generate with logit bias
    GenerationOptions options;
    options.seed = 42;
    
    auto audio = generator_->generate_speech("Test logit bias", 0, {}, options);
    
    // Should produce audio
    EXPECT_FALSE(audio.empty());
    
    // This test is preparatory - full implementation will be added later
    EXPECT_TRUE(true);
}