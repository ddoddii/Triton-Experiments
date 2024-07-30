# google-bert/bert-base-uncased

- [huggingface link](https://huggingface.co/google-bert/bert-base-uncased)
- params : 110M

## Task : Masked Language Modeling

<img width="585" alt="image" src="https://github.com/user-attachments/assets/fc35f03b-2107-41b0-83f2-0a4247a23ad4">

## Inference with triton server

### Experiments

1. **Default**

- concurrency = 4

    <img width="1230" alt="image" src="https://github.com/user-attachments/assets/9269db74-8777-4e86-a5ae-5ba5a9a4a872">


- concurrency = 10

    <img width="1253" alt="image" src="https://github.com/user-attachments/assets/463c0706-2882-4c21-bbe0-d8a8a4e8d400">

- concurrency = 16

    <img width="1261" alt="image" src="https://github.com/user-attachments/assets/79743ace-1281-4cda-b9bc-2961054c3784">

- Inferences/Second vs. Client p95 Batch Latency

    <img width="574" alt="image" src="https://github.com/user-attachments/assets/06fcc0d2-dd57-497d-8908-35894e1c350e">

- nsight

2.  **2 instance group, dynamic batching**

```text
instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]

dynamic_batching {}
```

- concurrency = 4

    <img width="1222" alt="image" src="https://github.com/user-attachments/assets/8d0edc40-763e-457e-af7e-1f917aaa2fd7">


- concurrency = 10

    <img width="1245" alt="image" src="https://github.com/user-attachments/assets/f87fcb6f-05e5-4dab-a5fa-1d83be16126e">

- concurrency = 16

    <img width="1253" alt="image" src="https://github.com/user-attachments/assets/53fb9f9f-e211-4bee-ab7b-ee50aafbabd5">

- Inferences/Second vs. Client p95 Batch Latency

    <img width="565" alt="image" src="https://github.com/user-attachments/assets/1e24f052-262d-4b78-ad63-afa873335e77">
