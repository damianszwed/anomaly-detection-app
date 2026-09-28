# Spring AI embedding anomaly demo

This example uses Spring AI and an Ollama embedding model without a vector
database. It embeds several normal product descriptions, calculates their
centroid, and treats cosine distance from that centroid as an anomaly score.

## Prerequisites

Start Ollama and download the embedding model:

```bash
ollama serve
ollama pull nomic-embed-text
```

If Ollama is already running, only the `ollama pull` command is needed.

## Run

```bash
mvn spring-boot:run
```

Expected output has the following shape (exact scores depend on the model):

```text
NORMAL   score=...  similarity=...  text=1/2 inch pneumatic control valve
ANOMALY  score=...  similarity=...  text=chocolate cake recipe with strawberries
```

Change `demo.anomaly-threshold` in `src/main/resources/application.properties`
to experiment. A real application should choose its threshold using labelled
validation data and a larger, representative set of normal examples.

The vectors live only in memory. A vector database would become useful when
the normal dataset is too large to scan directly or must be persisted and
searched efficiently.
