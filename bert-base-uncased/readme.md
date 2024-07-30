# google-bert/bert-base-uncased

- [huggingface link](https://huggingface.co/google-bert/bert-base-uncased)
- params : 110M

## Task : Masked Language Modeling

<img width="585" alt="image" src="https://github.com/user-attachments/assets/fc35f03b-2107-41b0-83f2-0a4247a23ad4">

## Inference with triton server

### Experiments

1.  **2 instance group, dynamic batching**

```text
instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]

dynamic_batching {}
```

- concurrency = 10

    <img width="1245" alt="image" src="https://github.com/user-attachments/assets/f87fcb6f-05e5-4dab-a5fa-1d83be16126e">

- Inferences/Second vs. Client p95 Batch Latency

    <img width="565" alt="image" src="https://github.com/user-attachments/assets/1e24f052-262d-4b78-ad63-afa873335e77">
