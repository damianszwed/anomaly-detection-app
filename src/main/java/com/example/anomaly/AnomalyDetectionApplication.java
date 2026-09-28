package com.example.anomaly;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class AnomalyDetectionApplication {

  public static void main(String[] args) {
    SpringApplication.run(AnomalyDetectionApplication.class, args);
  }

  @Bean
  CommandLineRunner anomalyDemo(
      EmbeddingModel embeddingModel,
      @Value("${demo.anomaly-threshold:0.35}") double threshold) {

    return args -> {
      // These examples define what "normal" means for this tiny demonstration.
      List<String> normalDescriptions = List.of(
          "24 volt DC pneumatic solenoid valve",
          "3/8 inch directional air control valve",
          "double solenoid four-way valve",
          "pneumatic valve repair kit",
          "foot-operated air valve");

      List<float[]> normalEmbeddings = new ArrayList<>();
      for (String description : normalDescriptions) {
        normalEmbeddings.add(embeddingModel.embed(description));
      }

      float[] normalCentroid = centroid(normalEmbeddings);

      List<String> observations = List.of(
          "1/2 inch pneumatic control valve",
          "replacement seal kit for an air valve",
          "chocolate cake recipe with strawberries",
          "wireless gaming keyboard with RGB lighting");

      System.out.printf("Anomaly threshold: %.3f%n%n", threshold);

      for (String observation : observations) {
        float[] embedding = embeddingModel.embed(observation);
        double similarity = cosineSimilarity(embedding, normalCentroid);
        double anomalyScore = 1.0 - similarity;
        boolean anomaly = anomalyScore > threshold;

        System.out.printf(
            "%s  score=%.3f  similarity=%.3f  text=%s%n",
            anomaly ? "ANOMALY" : "NORMAL ",
            anomalyScore,
            similarity,
            observation);
      }
    };
  }

  /**
   * Calculates the centroid of the supplied embedding vectors. A centroid is
   * the center, or average position, of a group of points. Here, every embedding
   * is a point representing a normal product description, so their centroid is
   * a single vector representing a "typical" normal description.
   *
   * <p>A new description can be compared with this centroid. If its embedding
   * is far from the centroid, it may be considered an anomaly.</p>
   *
   * <p>Each input vector is normalized before it is added to the centroid so
   * that vectors with larger magnitudes do not have greater influence. The
   * resulting sum is normalized again, making it suitable for cosine-similarity
   * comparisons with new embeddings.</p>
   *
   * @param vectors non-empty embedding vectors having identical dimensions
   * @return a new unit-length vector representing the center of the embeddings
   * @throws IllegalArgumentException if the list is empty, the dimensions differ,
   *     or any required normalization encounters a zero vector
   */
  static float[] centroid(List<float[]> vectors) {
    if (vectors.isEmpty()) {
      throw new IllegalArgumentException("At least one normal vector is required");
    }

    int dimensions = vectors.getFirst().length;
    float[] result = new float[dimensions];

    for (float[] vector : vectors) {
      if (vector.length != dimensions) {
        throw new IllegalArgumentException("All vectors must have the same dimensions");
      }
      float[] normalized = normalized(vector);
      for (int i = 0; i < dimensions; i++) {
        result[i] += normalized[i];
      }
    }

    return normalized(result);
  }

  static double cosineSimilarity(float[] left, float[] right) {
    if (left.length != right.length) {
      throw new IllegalArgumentException("Vectors must have the same dimensions");
    }

    double dotProduct = 0.0;
    double leftMagnitude = 0.0;
    double rightMagnitude = 0.0;

    for (int i = 0; i < left.length; i++) {
      dotProduct += left[i] * right[i];
      leftMagnitude += left[i] * left[i];
      rightMagnitude += right[i] * right[i];
    }

    if (leftMagnitude == 0.0 || rightMagnitude == 0.0) {
      throw new IllegalArgumentException("A zero vector has no cosine similarity");
    }

    return dotProduct / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude));
  }

  static float[] normalized(float[] vector) {
    double magnitudeSquared = 0.0;
    for (float value : vector) {
      magnitudeSquared += value * value;
    }

    if (magnitudeSquared == 0.0) {
      throw new IllegalArgumentException("Cannot normalize a zero vector");
    }

    double magnitude = Math.sqrt(magnitudeSquared);
    float[] result = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      result[i] = (float) (vector[i] / magnitude);
    }
    return result;
  }
}
