# BRL_project2_GDS

- 주관 : 과학기술정보통신부의 기초 연구실
- 주제 : "Descriptor search algorithm for material property prediction"
- 일시 : 2021.09 ~ 2022.09
- 내용 : DFT(density function theory), 또는 MD(molecular dynamics)를 활용하여 물성 예측을 하는데는 시간 및 계산량이 매우 많이 소요됨. 따라서 이를 머신러닝(descriptor search)으로 대체하고, 나아가 효율적으로 전역 최적해를 찾기 위해 유전 알고리즘을 활용함.


## Genetic descriptor search algorithm for predicting hydrogen adsorption free energy of 2D material
- Scientific Reports [In-preparation]
- First author

</br>

### 2D-TMD (Transition Metal Dichalcogenides) 데이터 셋을 활용하여 촉매 적합도를 나타내는 $\Delta G_H$를 예측하는 전체 frame work

<p align="center">
    <img src="./assets/Overview.png" width="90%" />
</p>

### 전체 알고리즘 수도코드

<p align="center">
    <img src="./assets/algorithm.PNG" width="70%" />
</p>

### 사용한 genetic operator

<p align="center">
    <img src="./assets/genetic_operators.PNG" width="90%" />
</p>

### Obtained descriptor가 실제 target의 manifold를 잘 나타낼 수 있다는 것을 정성적으로 확인.

<p align="center">
    <img src="./assets/TSNE.PNG" width="90%" />
</p>

### 2D descriptor가 찾은 hyperplane과 True

![2D example](https://user-images.githubusercontent.com/59224742/230302804-17f2cee8-eb87-4ce8-9f25-c5a06e7b8c91.gif)
