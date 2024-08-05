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


### Nsight Profiling 


**(터미널1)**
1.	docker 컨테이너 실행 (nsys 깔려 있는 tritonserver image)

```sh
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models tritonserver-with-nsys
```

2.	컨테이너 안에서 nsys profiling 실행

```sh
nsys profile --output=triton_profile -t cuda,osrt,nvtx,cudnn --export=sqlite -f true -w true tritonserver --model-repository=/models
```

<img width="682" alt="image" src="https://github.com/user-attachments/assets/321faf78-1f8e-4f68-b192-1e3fbac23a81">


**(터미널2)** 
3.	다른 터미널 키고 tritonserver:23.09-py3-sdk 이미지 실행

```sh
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.09-py3-sdk bash
```

4.	컨테이너 안에서 perf_analyzer 로 request 보내기 

```sh
perf_analyzer -m bert-base-uncased -b 2   --shape input_ids:128   --shape attention_mask:128   --shape token_type_ids:128   --concurrency-range 4:20:4 --percentile=95
```

(여기서 bert-base-uncased 는 triton 상에서 로드 된 모델 이름) 

<img width="584" alt="image" src="https://github.com/user-attachments/assets/078f69ed-1321-4e75-beb7-6aabfe8d6709">


**(터미널1)**

5.	ctrl+c 누르면 triton server 가 종료되면서 triton_profile.nsys-rep 가 생성됨 

<img width="1135" alt="image" src="https://github.com/user-attachments/assets/ab151755-0a90-4969-8c96-22b07f39dd78">


**(터미널3)** 

6.	docker ps 로 현재 실행 중인 docker container 보기 


<img width="1136" alt="image" src="https://github.com/user-attachments/assets/2a36a49b-516e-4811-9be0-ef0efe187038">


7.	docker container 안에 있는 report 를 원격 서버로 옮기기 

```sh
docker cp {container_id}:/opt/tritonserver/triton_profile.nsys-rep /home/uhmturks/triton-experiment/bert-base-uncased
```


