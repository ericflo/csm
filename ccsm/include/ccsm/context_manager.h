#ifndef CCSM_CONTEXT_MANAGER_H
#define CCSM_CONTEXT_MANAGER_H

#include <ccsm/tensor.h>
#include <ccsm/tokenizer.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace ccsm {

/**
 * Configuration for context management
 */
struct ContextConfig {
    // Maximum number of context tokens to keep
    int max_context_tokens = 2048;
    
    // Maximum memory size for the context in megabytes
    size_t max_context_memory_mb = 128;
    
    // Whether to use token pruning when the context gets too large
    bool enable_pruning = true;
    
    // Method for pruning when context gets too large
    enum class PruningStrategy {
        // Keep most recent tokens, discard oldest
        RECENCY,
        
        // Keep tokens with highest importance scores
        IMPORTANCE,
        
        // Keep a mix of important and recent tokens
        HYBRID
    };
    
    // The strategy to use for pruning
    PruningStrategy pruning_strategy = PruningStrategy::HYBRID;
    
    // Weight between recency and importance for HYBRID strategy (0=recency only, 1=importance only)
    float importance_weight = 0.5f;
    
    // Whether to automatically compress similar segments
    bool enable_segment_compression = false;
    
    // Similarity threshold for compression (0.0-1.0)
    float compression_similarity_threshold = 0.8f;
};

/**
 * Enhanced context segment with additional metadata
 */
struct EnhancedSegment : public Segment {
    // When this segment was added to the context
    uint64_t timestamp = 0;
    
    // Importance score for this segment (higher = more important)
    float importance_score = 1.0f;
    
    // Whether this segment has been compressed
    bool is_compressed = false;
    
    // Original text before compression (if compressed)
    std::string original_text;
    
    // Token indices that map to this segment
    std::vector<size_t> token_indices;
};

/**
 * Advanced context management system
 * 
 * This class provides enhanced context management capabilities for long conversations,
 * including dynamic context pruning, importance scoring, and segment compression.
 */
class ContextManager {
public:
    // Constructor
    ContextManager(std::shared_ptr<TextTokenizer> tokenizer);
    
    // Constructor with configuration
    ContextManager(std::shared_ptr<TextTokenizer> tokenizer, const ContextConfig& config);
    
    // Add a segment to the context
    void add_segment(const Segment& segment);
    
    // Add an enhanced segment to the context
    void add_enhanced_segment(const EnhancedSegment& segment);
    
    // Clear the context
    void clear();
    
    // Get the current context as a vector of tokens
    std::vector<int> get_context_tokens() const;
    
    // Get the current context positions
    std::vector<int> get_context_positions() const;
    
    // Get the current context as a vector of segments
    std::vector<Segment> get_context_segments() const;
    
    // Get the current context as a vector of enhanced segments
    const std::vector<EnhancedSegment>& get_enhanced_segments() const;
    
    // Get the total number of tokens in the context
    size_t get_token_count() const;
    
    // Estimate the memory usage of the context in bytes
    size_t estimate_memory_usage() const;
    
    // Prune the context to fit within limits
    bool prune_context();
    
    // Set the importance score for a segment
    bool set_segment_importance(size_t segment_index, float importance_score);
    
    // Set importance scores based on an external scoring function
    void set_importance_scores(const std::function<float(const EnhancedSegment&)>& scoring_function);
    
    // Compress similar segments to save context space
    bool compress_similar_segments();
    
    // Set the configuration
    void set_config(const ContextConfig& config);
    
    // Get the current configuration
    const ContextConfig& get_config() const;
    
private:
    // The tokenizer used for encoding/decoding
    std::shared_ptr<TextTokenizer> tokenizer_;
    
    // The context configuration
    ContextConfig config_;
    
    // The current context segments
    std::vector<EnhancedSegment> segments_;
    
    // Current token count in the context
    size_t token_count_ = 0;
    
    // Next timestamp value
    uint64_t next_timestamp_ = 0;
    
    // Private helper methods
    
    // Calculate similarity between two segments
    float calculate_segment_similarity(const EnhancedSegment& a, const EnhancedSegment& b) const;
    
    // Find segments to compress
    std::vector<std::pair<size_t, size_t>> find_compression_candidates() const;
    
    // Compress two segments
    EnhancedSegment compress_segments(const EnhancedSegment& a, const EnhancedSegment& b) const;
    
    // Calculate importance scores for tokens based on the pruning strategy
    std::vector<float> calculate_token_importance() const;
    
    // Rebuild token indices after context modification
    void rebuild_token_indices();
};

} // namespace ccsm

#endif // CCSM_CONTEXT_MANAGER_H