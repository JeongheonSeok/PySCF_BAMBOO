# PySCF_BAMBOO
Codes to generate dataset for training BAMBOO model

BAMBOO MLFF model:
Gong, Sheng, et al. "BAMBOO: a predictive and transferable machine learning force field framework for liquid electrolyte development." arXiv preprint arXiv:2404.07181 (2024). https://arxiv.org/abs/2404.07181
https://github.com/bytedance/bamboo

Caution!
해당 코드를 통해 생성한 데이터는 아직은 BAMBOO의 sample data를 완벽히 재현하지 못함. 아마 energy data baseline의 문제로 생각되나, 아직 명확한 해답을 찾지는 못한 상태. 아직은 사용에 주의를 요함. (2024/08/29)

학습용 DFT 데이터세트는 gpu4pyscf와 pyscf로 생성. ➡ pyscf 및 gpu4pyscf 및 관련 파이썬 패키지는 Linux 환경에서 설치 및 이용 가능. 
https://github.com/pyscf/pyscf
https://github.com/pyscf/gpu4pyscf

## 0. DFT 계산용 구조 생성
Classical MD를 수행하고, 이 중 일부 스냅샷의 랜덤 클러스터를 추출, 이를 데이터로 이용.

클러스터 구조는 .xyz 파일로 추출. 이때 파일명에 클러스터를 구성하는 이온 및 분자들의 이름과 개수가 포함되어 있어야 함. (ex: Frame10_0-7_sub6-0_TFDMP5FSI1.xyz)

## 1. DFT 계산
DFT 계산 조건: Restricted Kohn-Sham, B3LYP, def2-svpd, Density Fitting, auxiliary basis: def2-universal_jkfit, SCF convergence threshold 1.0 10^-10 a.u., maximum iteration 100.
1. 계산할 .xyz 파일들과 calc_from_xyzs.py, run_calc_from_xyzs.sh 파일들을 같은 디렉토리에 넣어둔다.
2. calc_from_xyzs.py 의 ION_LIST를 계산할 시스템에 맞게 적절히 수정한다.

   업로드된 파일은 LiFSI만이 이온인 경우로 저장되어 있다.
   
   이때 ion name은 .xyz파일에서 사용된 이름을 그대로 사용해야하고, ion list에 넣지 않은 물질 명은 중성 분자로 여겨진다.
   
   이를 정확히 입력하지 않을 시, DFT 계산에 오류가 발생할 수 있다.
   
   만약 파동함수 정보까지 얻고 싶다면, 81번 line에 주석 처리된 부분을 주석 해제하여 .molden 파일이 생성되도록 할 수 있다.
4. run_calc_from_xyzs.sh를 사용 환경에 맞게 적절히 수정한다. 해당 파일에는 가장 기본적인 내용만이 포함되어 있다.
5. 터미널에서 bash ./run_calc_from_xyzs.sh 를 하면 해당 폴더의 모든 xyz파일들에 대해 DFT계산이 이루어진다.
6. 계산 후에는 .xyz파일명과 동일한 이름의 .log 파일이 생성된다. 해당 파일은 cluster의 energy, gradient, dipole quadrupole moment, ChElPG charge 정보를 포함한다.

## 2. DFT 계산 결과를 BAMBOO 학습 데이터로 변환
DFT 계산 결과(.log 파일)들을 모아 .pt 파일로 변환해야 BAMBOO 모델 학습에 이용 가능하다.

log_to_zip.py: 각 cluster들의 DFT 계산 결과를 .pkl 파일에 저장하고, 이들을 모아 zip파일로 압축한다.

pkls_to_tvt.py: 각 cluster들의 정보가 담긴 .pkl파일들을 읽고, 이로부터 BAMBOO 학습용 데이터 .pt 파일을 만든다.
1. .log 파일들이 있는 폴더에 log_to_zip.py를 넣어둔다.
2. log_to_zip.py 의 ION_LIST를 계산한 시스템에 맞게 적절히 수정한다. 이는 calc_from_xyzs.py와 동일하다.
3. 이후 log_to_zip.py 실행 시, pkl.zip 파일이 생성된다.
4. 이렇게 생성한 모든 pkl 파일들을 모두 한 폴더에 넣는다.
5. pkls_to_tvt.py 파일에서 random seed, pkl 파일들이 모인 경로와 BAMBOO 학습용 .pt 파일이 생성될 경로, train val test ratio를 적절히 입력한다.
6. pkls_to_tvt.py를 실행하면 입력해둔 out_path에 train.pt, val.pt(+test.pt)가 생성된다.

## 3. BAMBOO 모델 학습
0. BAMBOO 모델을 다운받는다. (https://github.com/bytedance/bamboo)
1. BAMBOO 폴더에 data 폴더를 만들고, 해당 폴더에 위에서 생성한 train.pt, val.pt 파일을 넣는다.
2. BAMBOO 폴더 아래의 /configs/train_config/config.json 을 원하는 학습대로 적절히 수정한다.
3. 터미널 작업 디렉토리를 BAMBOO 폴더로 이동하고, python -m train.train --config configs/train_config/config.json 로 모델 학습을 진행한다.
4. 학습된 모델 파라미터 및 로그는 BAMBOO 폴더 아래의 train/bamboo_community_train/에 생긴다. checkpoint .pt들은 checkpoints 폴더에, train log는 train_logs 폴더에 생긴다.
