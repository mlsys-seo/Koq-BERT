# Koq-BERT

## 한글 BERT 모델에 Kernel fusion 및 FP16 양자화 적용

- 현재(2021년 1월) Saltlux AI홈페이지(https://www.saltlux.ai/portal/api_detail?id=category14) 에서 Kernel fusion 및 FP16 양자화를 적용한 모델을 서비스 중에 있습니다.
- 모델: BERT-Large
- 데이터셋: Saltlux에서 제공한 감성분석 evalutaion dataset(한 문장에 대한 극성(긍정, 부정, 중립) classification task)
- 두 문장간의 유사성을 판별하는 MRPC Task에 적용 가능

![model_img](https://user-images.githubusercontent.com/33375019/105456789-1d9de100-5cc9-11eb-9e75-e01404f0784d.png)

- 실험 환경
  - python : 3.5
  - tensorflow-gpu : 1.15.0
  - cmake : 3.8.2
  - CUDA : 10.0
  - cuDNN : 7.6.5
  - GPU : Tesla V100 32GB(Volta), RTX 2080ti 11GB(Turing)


- 실험 준비

  - src 폴더의 ckpt_type_convert.py 이용하여 모델의 파라미터를 FP16으로 양자화(convert.sh 참조)
  - Tensorflow에 CUDA kernel custom operation 등록

  ```shell
  $ export TF_DIR=/usr/local/lib/python3.5/dist-packages/tensorflow_core
  
  $ cd build_lib
  $ mkdir build
  $ cd build
  $ cmake -DSM=70 -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TF=ON -DTF_PATH=$TF_DIR ..
  $ ln -s $TF_DIR/libtensorflow_framework.so.1 $TF_DIR/libtensorflow_framework.so
  $ make tf_fastertransformer
  ```

  -DSM에는 사용하는 GPU에 해당하는 Compute Capability을 기입. ex) V100(70), RTX 2080ti(75)

  -DTF_PATH에는 로컬 경로의 tensorflow 패키지 경로를 기입.

   가상환경 anaconda를 사용시 tensorflow 패키지 경로는 아래와 같다.

    ex) /home/hyu/anaconda3/envs/salt/lib/python3.5/site-packages/tensorflow_core

  -부분 양자화 추가

  ​	-- FP16 부분 양자화 코드 추가(ckpt_type_convert2.py 및 convert2.sh 참조) 

  ​	-- 9 ~ 24번째 Transformer encoder를 FP16으로 부분 양자화 적용

  ​	-- 부분 양자화를 통해 추론 시간 향상 및 정확도 유지

  

- 실행 방법

  ```shell
  $ python classifier.py --floatx float16 --batch_size 8
  ```

  -기본적으로 모델에 Kernel fusion을 적용하였으며, argument를 통해 float32, float16, mix를 선택할 수 있다.

  -batch_size를 입력하여 사용할 수 있으며, 마지막 iteration의 입력 데이터 수가 batch size보다 작을 경우 drop.

  -부분양자화 모델을 사용할 경우 floatx를 mix로 사용.

  

- 성능 측정 결과

  ![table_img](https://user-images.githubusercontent.com/33375019/105458787-8a66aa80-5ccc-11eb-8c36-abb7074a8fff.png)
  
 
