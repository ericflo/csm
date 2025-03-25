#include <ccsm/context_manager.h>
#include <ccsm/utils.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <sstream>
#include <unordered_set>

namespace ccsm {

// Constructor
ContextManager::ContextManager(std::shared_ptr<TextTokenizer> tokenizer)
    : tokenizer_(tokenizer), token_count_(0), next_timestamp_(0) {
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
}

// Constructor with configuration
ContextManager::ContextManager(std::shared_ptr<TextTokenizer> tokenizer, const ContextConfig& config)
    : tokenizer_(tokenizer), config_(config), token_count_(0), next_timestamp_(0) {
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
}

// Add a segment to the context
void ContextManager::add_segment(const Segment& segment) {
    // Create enhanced segment with base constructor first
    EnhancedSegment enhanced(segment.text, segment.speaker_id, segment.audio);
    enhanced.timestamp = next_timestamp_++;
    enhanced.importance_score = 1.0f; // Default importance
    enhanced.is_compressed = false;
    
    add_enhanced_segment(enhanced);
}

// Add an enhanced segment to the context
void ContextManager::add_enhanced_segment(const EnhancedSegment& segment) {
    // Get tokens for this segment
    std::vector<int> segment_tokens;
    
    // Add speaker token if provided
    if (segment.speaker_id >= 0) {
        segment_tokens.push_back(tokenizer_->get_speaker_token_id(segment.speaker_id));
    }
    
    // Add text tokens
    if (!segment.text.empty()) {
        std::vector<int> text_tokens = tokenizer_->encode(segment.text);
        segment_tokens.insert(segment_tokens.end(), text_tokens.begin(), text_tokens.end());
    }
    
    // Update token count
    token_count_ += segment_tokens.size();
    
    // Add segment with token indices
    EnhancedSegment new_segment = segment;
    
    // Update token indices
    size_t start_idx = 0;
    for (const auto& existing_segment : segments_) {
        start_idx += existing_segment.token_indices.size();
    }
    
    // Set token indices for the new segment
    new_segment.token_indices.resize(segment_tokens.size());
    for (size_t i = 0; i < segment_tokens.size(); i++) {
        new_segment.token_indices[i] = start_idx + i;
    }
    
    // Add to segments
    segments_.push_back(new_segment);
    
    // If we're over the limits, prune the context
    if ((token_count_ > static_cast<size_t>(config_.max_context_tokens)) ||
        (estimate_memory_usage() > config_.max_context_memory_mb * 1024 * 1024)) {
        
        if (config_.enable_pruning) {
            prune_context();
        }
    }
    
    // If compression is enabled, try to compress similar segments
    if (config_.enable_segment_compression && segments_.size() > 1) {
        compress_similar_segments();
    }
}

// Clear the context
void ContextManager::clear() {
    segments_.clear();
    token_count_ = 0;
}

// Get the current context as a vector of tokens
std::vector<int> ContextManager::get_context_tokens() const {
    std::vector<int> tokens;
    tokens.reserve(token_count_);
    
    for (const auto& segment : segments_) {
        // Add speaker token if present
        if (segment.speaker_id >= 0) {
            tokens.push_back(tokenizer_->get_speaker_token_id(segment.speaker_id));
        }
        
        // Add text tokens
        std::vector<int> text_tokens = tokenizer_->encode(segment.text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    }
    
    return tokens;
}

// Get the current context positions
std::vector<int> ContextManager::get_context_positions() const {
    std::vector<int> positions;
    positions.reserve(token_count_);
    
    for (size_t i = 0; i < token_count_; i++) {
        positions.push_back(static_cast<int>(i));
    }
    
    return positions;
}

// Get the current context as a vector of segments
std::vector<Segment> ContextManager::get_context_segments() const {
    std::vector<Segment> segments;
    segments.reserve(segments_.size());
    
    for (const auto& enhanced_segment : segments_) {
        // Create segment with proper constructor
        Segment segment(enhanced_segment.text, enhanced_segment.speaker_id, enhanced_segment.audio);
        segments.push_back(segment);
    }
    
    return segments;
}

// Get the current context as a vector of enhanced segments
const std::vector<EnhancedSegment>& ContextManager::get_enhanced_segments() const {
    return segments_;
}

// Get the total number of tokens in the context
size_t ContextManager::get_token_count() const {
    return token_count_;
}

// Estimate the memory usage of the context in bytes
size_t ContextManager::estimate_memory_usage() const {
    size_t memory_usage = 0;
    
    // Context tokens memory (assuming int is 4 bytes)
    memory_usage += token_count_ * sizeof(int);
    
    // Enhanced segments memory
    for (const auto& segment : segments_) {
        // Text storage
        memory_usage += segment.text.size() * sizeof(char);
        
        // Original text if compressed
        if (segment.is_compressed) {
            memory_usage += segment.original_text.size() * sizeof(char);
        }
        
        // Token indices
        memory_usage += segment.token_indices.size() * sizeof(size_t);
        
        // Other segment data
        memory_usage += sizeof(EnhancedSegment) - sizeof(std::string) - sizeof(std::vector<size_t>);
    }
    
    return memory_usage;
}

// Prune the context to fit within limits
bool ContextManager::prune_context() {
    if (segments_.empty()) {
        return false; // Nothing to prune
    }
    
    CCSM_INFO("Pruning context from ", token_count_, " tokens (", segments_.size(), " segments)");
    
    // Calculate target token count (80% of max to avoid frequent pruning)
    size_t target_tokens = static_cast<size_t>(config_.max_context_tokens * 0.8);
    
    // If we're not over the limit, no pruning needed
    if (token_count_ <= target_tokens) {
        return false;
    }
    
    // Calculate how many tokens to remove
    size_t tokens_to_remove = token_count_ - target_tokens;
    
    // Different pruning strategies
    switch (config_.pruning_strategy) {
        case ContextConfig::PruningStrategy::RECENCY: {
            // Remove oldest segments first
            std::vector<EnhancedSegment> pruned_segments;
            size_t tokens_removed = 0;
            
            // Sort segments by timestamp (oldest first)
            std::vector<EnhancedSegment> sorted_segments = segments_;
            std::sort(sorted_segments.begin(), sorted_segments.end(),
                      [](const EnhancedSegment& a, const EnhancedSegment& b) {
                          return a.timestamp < b.timestamp;
                      });
            
            // Remove oldest segments until we've removed enough tokens
            for (size_t i = 0; i < sorted_segments.size(); i++) {
                const auto& segment = sorted_segments[i];
                size_t segment_tokens = segment.token_indices.size();
                
                if (tokens_removed + segment_tokens <= tokens_to_remove) {
                    // Remove this segment
                    tokens_removed += segment_tokens;
                } else {
                    // Keep this segment
                    pruned_segments.push_back(segment);
                }
                
                // If we've removed enough tokens, keep all remaining segments
                if (tokens_removed >= tokens_to_remove) {
                    pruned_segments.insert(pruned_segments.end(),
                                          sorted_segments.begin() + i + 1,
                                          sorted_segments.end());
                    break;
                }
            }
            
            // Update segments and token count
            segments_ = pruned_segments;
            token_count_ -= tokens_removed;
            
            // Rebuild token indices
            rebuild_token_indices();
            break;
        }
        
        case ContextConfig::PruningStrategy::IMPORTANCE: {
            // Remove least important segments first
            std::vector<EnhancedSegment> pruned_segments;
            size_t tokens_removed = 0;
            
            // Sort segments by importance (least important first)
            std::vector<EnhancedSegment> sorted_segments = segments_;
            std::sort(sorted_segments.begin(), sorted_segments.end(),
                      [](const EnhancedSegment& a, const EnhancedSegment& b) {
                          return a.importance_score < b.importance_score;
                      });
            
            // Remove least important segments until we've removed enough tokens
            for (size_t i = 0; i < sorted_segments.size(); i++) {
                const auto& segment = sorted_segments[i];
                size_t segment_tokens = segment.token_indices.size();
                
                if (tokens_removed + segment_tokens <= tokens_to_remove) {
                    // Remove this segment
                    tokens_removed += segment_tokens;
                } else {
                    // Keep this segment
                    pruned_segments.push_back(segment);
                }
                
                // If we've removed enough tokens, keep all remaining segments
                if (tokens_removed >= tokens_to_remove) {
                    pruned_segments.insert(pruned_segments.end(),
                                          sorted_segments.begin() + i + 1,
                                          sorted_segments.end());
                    break;
                }
            }
            
            // Update segments and token count
            segments_ = pruned_segments;
            token_count_ -= tokens_removed;
            
            // Rebuild token indices
            rebuild_token_indices();
            break;
        }
        
        case ContextConfig::PruningStrategy::HYBRID: {
            // Use both recency and importance
            std::vector<EnhancedSegment> pruned_segments;
            size_t tokens_removed = 0;
            
            // Calculate hybrid scores (combination of recency and importance)
            std::vector<std::pair<size_t, float>> segment_scores;
            for (size_t i = 0; i < segments_.size(); i++) {
                const auto& segment = segments_[i];
                
                // Normalize timestamp to 0-1 range
                float recency_score = 1.0f - static_cast<float>(segment.timestamp) / static_cast<float>(next_timestamp_);
                
                // Combine recency and importance
                float hybrid_score = (1.0f - config_.importance_weight) * recency_score + 
                                     config_.importance_weight * segment.importance_score;
                                     
                segment_scores.push_back({i, hybrid_score});
            }
            
            // Sort by hybrid score (lowest first)
            std::sort(segment_scores.begin(), segment_scores.end(),
                      [](const auto& a, const auto& b) {
                          return a.second < b.second;
                      });
            
            // Mark segments to keep
            std::vector<bool> keep_segment(segments_.size(), true);
            
            // Remove lowest-scored segments until we've removed enough tokens
            for (const auto& [idx, score] : segment_scores) {
                const auto& segment = segments_[idx];
                size_t segment_tokens = segment.token_indices.size();
                
                if (tokens_removed + segment_tokens <= tokens_to_remove) {
                    // Remove this segment
                    keep_segment[idx] = false;
                    tokens_removed += segment_tokens;
                }
                
                // If we've removed enough tokens, stop
                if (tokens_removed >= tokens_to_remove) {
                    break;
                }
            }
            
            // Build pruned segments list
            for (size_t i = 0; i < segments_.size(); i++) {
                if (keep_segment[i]) {
                    pruned_segments.push_back(segments_[i]);
                }
            }
            
            // Update segments and token count
            segments_ = pruned_segments;
            token_count_ -= tokens_removed;
            
            // Rebuild token indices
            rebuild_token_indices();
            break;
        }
    }
    
    CCSM_INFO("Pruned context to ", token_count_, " tokens (", segments_.size(), " segments)");
    return true;
}

// Set the importance score for a segment
bool ContextManager::set_segment_importance(size_t segment_index, float importance_score) {
    if (segment_index >= segments_.size()) {
        return false;
    }
    
    segments_[segment_index].importance_score = std::max(0.0f, std::min(importance_score, 1.0f));
    return true;
}

// Set importance scores based on an external scoring function
void ContextManager::set_importance_scores(const std::function<float(const EnhancedSegment&)>& scoring_function) {
    for (auto& segment : segments_) {
        float score = scoring_function(segment);
        segment.importance_score = std::max(0.0f, std::min(score, 1.0f));
    }
}

// Compress similar segments to save context space
bool ContextManager::compress_similar_segments() {
    if (!config_.enable_segment_compression || segments_.size() < 2) {
        return false;
    }
    
    // Find candidates for compression
    auto compression_candidates = find_compression_candidates();
    if (compression_candidates.empty()) {
        return false;
    }
    
    CCSM_INFO("Compressing ", compression_candidates.size(), " segment pairs");
    
    // Track changes in token count for proper updating
    size_t original_token_count = token_count_;
    
    // Process each pair (from highest indices to lowest to avoid invalidation)
    std::sort(compression_candidates.begin(), compression_candidates.end(),
              [](const auto& a, const auto& b) {
                  return std::max(a.first, a.second) > std::max(b.first, b.second);
              });
    
    // Track which segments have been compressed already
    std::unordered_set<size_t> compressed_indices;
    
    // Compress pairs
    std::vector<EnhancedSegment> new_segments;
    for (const auto& [idx1, idx2] : compression_candidates) {
        // Skip if either segment has already been compressed in another pair
        if (compressed_indices.count(idx1) > 0 || compressed_indices.count(idx2) > 0) {
            continue;
        }
        
        // Mark these segments as compressed
        compressed_indices.insert(idx1);
        compressed_indices.insert(idx2);
        
        // Compress the segments
        EnhancedSegment compressed = compress_segments(segments_[idx1], segments_[idx2]);
        new_segments.push_back(compressed);
    }
    
    // Add uncompressed segments
    for (size_t i = 0; i < segments_.size(); i++) {
        if (compressed_indices.count(i) == 0) {
            new_segments.push_back(segments_[i]);
        }
    }
    
    // Update segments
    segments_ = new_segments;
    
    // Recalculate token count
    token_count_ = 0;
    for (const auto& segment : segments_) {
        token_count_ += segment.token_indices.size();
    }
    
    // Rebuild token indices
    rebuild_token_indices();
    
    CCSM_INFO("Compressed from ", original_token_count, " to ", token_count_, " tokens");
    return original_token_count != token_count_;
}

// Set the configuration
void ContextManager::set_config(const ContextConfig& config) {
    config_ = config;
    
    // Apply changes if needed
    if (token_count_ > static_cast<size_t>(config_.max_context_tokens) && config_.enable_pruning) {
        prune_context();
    }
}

// Get the current configuration
const ContextConfig& ContextManager::get_config() const {
    return config_;
}

// Private helper methods

// Calculate similarity between two segments
float ContextManager::calculate_segment_similarity(const EnhancedSegment& a, const EnhancedSegment& b) const {
    // Skip if different speakers
    if (a.speaker_id != b.speaker_id) {
        return 0.0f;
    }
    
    // Skip if timestamps are too far apart
    if (std::abs(static_cast<int64_t>(a.timestamp) - static_cast<int64_t>(b.timestamp)) > 10) {
        return 0.0f;
    }
    
    // Encode each segment
    std::vector<int> tokens_a = tokenizer_->encode(a.text);
    std::vector<int> tokens_b = tokenizer_->encode(b.text);
    
    // If either set of tokens is empty, no similarity
    if (tokens_a.empty() || tokens_b.empty()) {
        return 0.0f;
    }
    
    // Calculate Jaccard similarity
    std::unordered_set<int> set_a(tokens_a.begin(), tokens_a.end());
    std::unordered_set<int> set_b(tokens_b.begin(), tokens_b.end());
    
    std::unordered_set<int> intersection;
    for (int token : set_a) {
        if (set_b.count(token) > 0) {
            intersection.insert(token);
        }
    }
    
    std::unordered_set<int> union_set;
    union_set.insert(set_a.begin(), set_a.end());
    union_set.insert(set_b.begin(), set_b.end());
    
    return static_cast<float>(intersection.size()) / static_cast<float>(union_set.size());
}

// Find segments to compress
std::vector<std::pair<size_t, size_t>> ContextManager::find_compression_candidates() const {
    std::vector<std::pair<size_t, size_t>> candidates;
    
    // Check similarity between each pair of segments
    for (size_t i = 0; i < segments_.size(); i++) {
        for (size_t j = i + 1; j < segments_.size(); j++) {
            float similarity = calculate_segment_similarity(segments_[i], segments_[j]);
            
            // If similarity is above threshold, add to candidates
            if (similarity >= config_.compression_similarity_threshold) {
                candidates.push_back({i, j});
            }
        }
    }
    
    return candidates;
}

// Compress two segments
EnhancedSegment ContextManager::compress_segments(const EnhancedSegment& a, const EnhancedSegment& b) const {
    // Create a temporary merged text for the base constructor
    std::string merged_text = a.text + " " + b.text;
    
    // Create with base constructor first
    EnhancedSegment merged(merged_text, a.speaker_id, {}); // Empty audio for now
    
    // Use the later timestamp
    merged.timestamp = std::max(a.timestamp, b.timestamp);
    
    // Use the higher importance score
    merged.importance_score = std::max(a.importance_score, b.importance_score);
    
    // Mark as compressed
    merged.is_compressed = true;
    
    // Combine original texts
    if (a.is_compressed && b.is_compressed) {
        merged.original_text = a.original_text + "\n" + b.original_text;
    } else if (a.is_compressed) {
        merged.original_text = a.original_text + "\n" + b.text;
    } else if (b.is_compressed) {
        merged.original_text = a.text + "\n" + b.original_text;
    } else {
        merged.original_text = a.text + "\n" + b.text;
    }
    
    // Create a compressed text that summarizes both segments
    // This is a simple approach - in a real implementation, you might use
    // a more sophisticated summarization method
    merged.text = a.text + " " + b.text;
    
    // Simple length limit to ensure compression
    const size_t max_length = std::min(a.text.length() + b.text.length() - 10, // Aim for some compression
                                      static_cast<size_t>(200)); // Hard cap
    
    if (merged.text.length() > max_length) {
        merged.text = merged.text.substr(0, max_length) + "...";
    }
    
    // Placeholder token indices that will be rebuilt
    merged.token_indices = {};
    
    return merged;
}

// Rebuild token indices after context modification
void ContextManager::rebuild_token_indices() {
    size_t index = 0;
    for (auto& segment : segments_) {
        // Calculate number of tokens in this segment
        size_t num_tokens = 0;
        
        // Add speaker token if present
        if (segment.speaker_id >= 0) {
            num_tokens++;
        }
        
        // Add text tokens
        std::vector<int> text_tokens = tokenizer_->encode(segment.text);
        num_tokens += text_tokens.size();
        
        // Update token indices
        segment.token_indices.resize(num_tokens);
        for (size_t i = 0; i < num_tokens; i++) {
            segment.token_indices[i] = index++;
        }
    }
}

} // namespace ccsm