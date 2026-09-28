package com.example.anomaly;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.Test;

class AnomalyDetectionApplicationTest {

  private static final float FLOAT_TOLERANCE = 1.0e-6f;
  private static final double DOUBLE_TOLERANCE = 1.0e-12;

  @Test
  void centroidAveragesVectorDirectionsAndReturnsAUnitVector() {
    float[] centroid = AnomalyDetectionApplication.centroid(List.of(
        new float[] {2.0f, 0.0f},
        new float[] {0.0f, 10.0f}));

    float expectedComponent = (float) (1.0 / Math.sqrt(2.0));
    assertArrayEquals(
        new float[] {expectedComponent, expectedComponent}, centroid, FLOAT_TOLERANCE);
    assertEquals(1.0, magnitude(centroid), FLOAT_TOLERANCE);
  }

  @Test
  void centroidOfOneVectorIsItsNormalizedCopy() {
    float[] vector = {3.0f, 4.0f};

    float[] centroid = AnomalyDetectionApplication.centroid(List.of(vector));

    assertArrayEquals(new float[] {0.6f, 0.8f}, centroid, FLOAT_TOLERANCE);
    assertNotSame(vector, centroid);
    assertArrayEquals(new float[] {3.0f, 4.0f}, vector);
  }

  @Test
  void centroidRejectsAnEmptyList() {
    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.centroid(List.of()));
  }

  @Test
  void centroidRejectsVectorsWithDifferentDimensions() {
    List<float[]> vectors = List.of(new float[] {1.0f, 2.0f}, new float[] {1.0f});

    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.centroid(vectors));
  }

  @Test
  void centroidRejectsAZeroInputVector() {
    List<float[]> vectors = List.of(new float[] {1.0f, 0.0f}, new float[] {0.0f, 0.0f});

    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.centroid(vectors));
  }

  @Test
  void centroidRejectsDirectionsThatCancelEachOther() {
    List<float[]> vectors = List.of(new float[] {1.0f, 0.0f}, new float[] {-1.0f, 0.0f});

    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.centroid(vectors));
  }

  @Test
  void cosineSimilarityIsOneForVectorsInTheSameDirection() {
    double similarity = AnomalyDetectionApplication.cosineSimilarity(
        new float[] {1.0f, 2.0f, 3.0f},
        new float[] {2.0f, 4.0f, 6.0f});

    assertEquals(1.0, similarity, DOUBLE_TOLERANCE);
  }

  @Test
  void cosineSimilarityIsZeroForOrthogonalVectors() {
    double similarity = AnomalyDetectionApplication.cosineSimilarity(
        new float[] {1.0f, 0.0f},
        new float[] {0.0f, -5.0f});

    assertEquals(0.0, similarity, DOUBLE_TOLERANCE);
  }

  @Test
  void cosineSimilarityIsMinusOneForOppositeVectors() {
    double similarity = AnomalyDetectionApplication.cosineSimilarity(
        new float[] {1.0f, -2.0f},
        new float[] {-3.0f, 6.0f});

    assertEquals(-1.0, similarity, DOUBLE_TOLERANCE);
  }

  @Test
  void cosineSimilarityRejectsVectorsWithDifferentDimensions() {
    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.cosineSimilarity(
            new float[] {1.0f}, new float[] {1.0f, 2.0f}));
  }

  @Test
  void cosineSimilarityRejectsAZeroVector() {
    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.cosineSimilarity(
            new float[] {0.0f, 0.0f}, new float[] {1.0f, 2.0f}));
  }

  @Test
  void normalizedReturnsAUnitVectorWithoutChangingItsInput() {
    float[] vector = {-3.0f, 4.0f};

    float[] normalized = AnomalyDetectionApplication.normalized(vector);

    assertArrayEquals(new float[] {-0.6f, 0.8f}, normalized, FLOAT_TOLERANCE);
    assertEquals(1.0, magnitude(normalized), FLOAT_TOLERANCE);
    assertNotSame(vector, normalized);
    assertArrayEquals(new float[] {-3.0f, 4.0f}, vector);
  }

  @Test
  void normalizedRejectsAZeroVector() {
    assertThrows(
        IllegalArgumentException.class,
        () -> AnomalyDetectionApplication.normalized(new float[] {0.0f, 0.0f}));
  }

  private static double magnitude(float[] vector) {
    double magnitudeSquared = 0.0;
    for (float value : vector) {
      magnitudeSquared += value * value;
    }
    return Math.sqrt(magnitudeSquared);
  }
}
