package com.example.semantic;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Production-ready semantic text splitter using TensorFlow Java.
 * Splits text into semantically coherent chunks using sentence transformers.
 * 
 * @author Production Team
 * @version 1.0
 */
@Component
public class SemanticTextSplitter implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(SemanticTextSplitter.class);
    
    private static final int DEFAULT_CHUNK_SIZE = 1000;
    private static final int DEFAULT_OVERLAP = 200;
    private static final double SIMILARITY_THRESHOLD = 0.7;
    
    private final SavedModelBundle model;
    private final Map<String, float[]> embeddingCache = new ConcurrentHashMap<>();
    private final int chunkSize;
    private final int overlapSize;

    /**
     * Creates semantic text splitter with default configuration.
     */
    public SemanticTextSplitter(Path modelPath) throws IOException {
        this(modelPath, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP);
    }

    /**
     * Creates semantic text splitter with custom chunk settings.
     */
    public SemanticTextSplitter(Path modelPath, int chunkSize, int overlapSize) throws IOException {
        this.chunkSize = chunkSize;
        this.overlapSize = overlapSize;
        
        try {
            this.model = SavedModelBundle.load(modelPath.toString(), "serve");
            log.info("Loaded TensorFlow model: {}", modelPath);
        } catch (Exception e) {
            log.error("Failed to load model: {}", modelPath, e);
            throw new IOException("Model loading failed", e);
        }
    }

    /**
     * Splits text into semantic chunks with content cleaning.
     * 
     * @param text Input text to split
     * @return List of cleaned text chunks as strings
     */
    public List<String> splitText(String text) {
        if (text == null || text.trim().isEmpty()) {
            return Collections.emptyList();
        }

        log.debug("Splitting text of {} characters", text.length());
        
        try {
            // Clean text before processing
            String cleanedText = cleanText(text);
            List<String> sentences = splitIntoSentences(cleanedText);
            Map<String, float[]> embeddings = computeEmbeddings(sentences);
            List<String> chunks = createSemanticChunks(sentences, embeddings);
            
            // Apply final cleaning to chunks
            return chunks.stream()
                    .map(this::finalCleanChunk)
                    .filter(chunk -> !chunk.isEmpty())
                    .collect(Collectors.toList());
            
        } catch (Exception e) {
            log.error("Text splitting failed", e);
            // Fallback to simple splitting with cleaning
            return fallbackSplit(cleanText(text));
        }
    }

    /**
     * Async text splitting for large documents.
     */
    public CompletableFuture<List<String>> splitTextAsync(String text) {
        return CompletableFuture.supplyAsync(() -> splitText(text));
    }

    /**
     * Comprehensive text cleaning for preprocessing.
     */
    private String cleanText(String text) {
        if (text == null) return "";
        
        return text
                // Normalize Unicode characters
                .replace('\u00A0', ' ')  // Non-breaking space
                .replace('\u2009', ' ')  // Thin space
                .replace('\u200B', ' ')  // Zero-width space
                .replace('\u2060', ' ')  // Word joiner
                .replace('\uFEFF', ' ')  // Zero-width no-break space (BOM)
                
                // Handle various quote types
                .replaceAll("[""]", "\"")
                .replaceAll("['']", "'")
                
                // Handle various dash types
                .replaceAll("[–—]", "-")
                
                // Handle ellipsis
                .replace("…", "...")
                
                // Remove control characters except tabs and newlines
                .replaceAll("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]", "")
                
                // Normalize multiple whitespace (spaces, tabs, newlines)
                .replaceAll("\\s+", " ")
                
                // Remove HTML/XML tags if present
                .replaceAll("<[^>]+>", " ")
                
                // Remove markdown formatting
                .replaceAll("\\*\\*([^*]+)\\*\\*", "$1")  // Bold
                .replaceAll("\\*([^*]+)\\*", "$1")        // Italic
                .replaceAll("__([^_]+)__", "$1")          // Bold underscore
                .replaceAll("_([^_]+)_", "$1")            // Italic underscore
                .replaceAll("`([^`]+)`", "$1")            // Inline code
                .replaceAll("\\[([^\\]]+)\\]\\([^)]+\\)", "$1")  // Links
                
                // Clean up extra spaces and trim
                .replaceAll("\\s+", " ")
                .trim();
    }

    /**
     * Final cleaning for individual chunks.
     */
    private String finalCleanChunk(String chunk) {
        if (chunk == null) return "";
        
        return chunk
                // Remove leading/trailing punctuation marks that don't make sense
                .replaceAll("^[,;:.!?\\-_]+", "")
                .replaceAll("[,;:\\-_]+$", "")
                
                // Fix spacing around punctuation
                .replaceAll("\\s*([.!?])\\s*", "$1 ")
                .replaceAll("\\s*([,;:])\\s*", "$1 ")
                
                // Remove duplicate punctuation
                .replaceAll("([.!?])\\1+", "$1")
                .replaceAll("([,;:])\\1+", "$1")
                
                // Fix quotation marks spacing
                .replaceAll("\\s*\"\\s*([^\"]+)\\s*\"\\s*", " \"$1\" ")
                .replaceAll("\\s*'\\s*([^']+)\\s*'\\s*", " '$1' ")
                
                // Remove excessive line breaks within chunk
                .replaceAll("\\n\\s*\\n", " ")
                
                // Final whitespace normalization
                .replaceAll("\\s+", " ")
                .trim();
    }

    /**
     * Compute embeddings for sentences with caching.
     */
    private Map<String, float[]> computeEmbeddings(List<String> sentences) {
        return sentences.parallelStream()
                .collect(Collectors.toConcurrentMap(
                    sentence -> sentence,
                    sentence -> embeddingCache.computeIfAbsent(sentence, this::computeEmbedding)
                ));
    }

    /**
     * Compute single sentence embedding using TensorFlow.
     */
    private float[] computeEmbedding(String sentence) {
        try (Session session = new Session(model.graph());
             TString input = TString.tensorOfBytes(Shape.of(1), 
                     data -> data.setObject(sentence.getBytes(), 0))) {
            
            List<Tensor> outputs = session.runner()
                    .feed("serving_default_input", input)
                    .fetch("StatefulPartitionedCall")
                    .run();
            
            try (TFloat32 output = (TFloat32) outputs.get(0)) {
                float[] embedding = new float[(int) output.size()];
                output.asRawTensor().data().asFloats().read(embedding);
                return normalizeVector(embedding);
            }
        } catch (Exception e) {
            log.warn("Embedding failed for sentence: {}", sentence.substring(0, Math.min(50, sentence.length())));
            return new float[384]; // Default embedding size
        }
    }

    /**
     * Normalize embedding vector.
     */
    private float[] normalizeVector(float[] vector) {
        double norm = Math.sqrt(Arrays.stream(vector)
                .asDoubleStream()
                .map(x -> x * x)
                .sum());
        
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] = (float) (vector[i] / norm);
            }
        }
        return vector;
    }

    /**
     * Create semantic chunks based on similarity scores.
     */
    private List<String> createSemanticChunks(List<String> sentences, Map<String, float[]> embeddings) {
        List<String> chunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();
        int currentLength = 0;

        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i);
            
            // Check if we need to split
            if (currentLength + sentence.length() > chunkSize && currentLength > 0) {
                // Check semantic similarity with next sentence
                if (i < sentences.size() - 1 && shouldSplitHere(embeddings, sentences, i)) {
                    chunks.add(currentChunk.toString().trim());
                    
                    // Add overlap
                    currentChunk = new StringBuilder(getOverlapText(currentChunk.toString()));
                    currentLength = currentChunk.length();
                }
            }

            // Add sentence to current chunk
            if (currentChunk.length() > 0) currentChunk.append(" ");
            currentChunk.append(sentence);
            currentLength += sentence.length() + 1;
        }

        // Add final chunk
        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString().trim());
        }

        return chunks;
    }

    /**
     * Determine if we should split at current position based on similarity.
     */
    private boolean shouldSplitHere(Map<String, float[]> embeddings, List<String> sentences, int index) {
        if (index >= sentences.size() - 1) return true;
        
        float[] current = embeddings.get(sentences.get(index));
        float[] next = embeddings.get(sentences.get(index + 1));
        
        if (current == null || next == null) return true;
        
        double similarity = cosineSimilarity(current, next);
        return similarity < SIMILARITY_THRESHOLD;
    }

    /**
     * Calculate cosine similarity between two vectors.
     */
    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) return 0.0;
        
        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        double denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator != 0.0 ? dotProduct / denominator : 0.0;
    }

    /**
     * Get overlap text from end of chunk.
     */
    private String getOverlapText(String text) {
        if (text.length() <= overlapSize) return text;
        
        String overlap = text.substring(text.length() - overlapSize);
        int spaceIndex = overlap.indexOf(' ');
        return spaceIndex > 0 ? overlap.substring(spaceIndex + 1) : overlap;
    }

    /**
     * Split cleaned text into sentences using enhanced regex.
     */
    private List<String> splitIntoSentences(String text) {
        return Arrays.stream(text.split("(?<=[.!?])\\s+"))
                .filter(s -> !s.trim().isEmpty())
                .map(this::cleanSentence)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }

    /**
     * Clean individual sentences.
     */
    private String cleanSentence(String sentence) {
        return sentence
                .trim()
                // Remove leading conjunctions that don't make sense at start
                .replaceAll("^(and|but|or|so)\\s+", "")
                // Ensure proper sentence ending
                .replaceAll("([^.!?])$", "$1.")
                // Clean multiple spaces
                .replaceAll("\\s+", " ")
                .trim();
    }

    /**
     * Fallback to simple text splitting with cleaning if semantic analysis fails.
     */
    private List<String> fallbackSplit(String text) {
        List<String> chunks = new ArrayList<>();
        int start = 0;
        
        while (start < text.length()) {
            int end = Math.min(start + chunkSize, text.length());
            
            // Try to end at sentence boundary first
            if (end < text.length()) {
                int lastSentence = Math.max(
                    text.lastIndexOf('.', end),
                    Math.max(text.lastIndexOf('!', end), text.lastIndexOf('?', end))
                );
                if (lastSentence > start) {
                    end = lastSentence + 1;
                } else {
                    // Fall back to word boundary
                    int lastSpace = text.lastIndexOf(' ', end);
                    if (lastSpace > start) end = lastSpace;
                }
            }
            
            String chunk = finalCleanChunk(text.substring(start, end));
            if (!chunk.isEmpty()) {
                chunks.add(chunk);
            }
            
            start = Math.max(start + chunkSize - overlapSize, end);
        }
        
        return chunks;
    }

    /**
     * Get cache statistics.
     */
    public Map<String, Integer> getCacheStats() {
        return Map.of("cache_size", embeddingCache.size());
    }

    @Override
    public void close() {
        try {
            if (model != null) {
                model.close();
                log.info("Model closed successfully");
            }
            clearCache();
        } catch (Exception e) {
            log.error("Error closing resources", e);
        }
    }
}
}
