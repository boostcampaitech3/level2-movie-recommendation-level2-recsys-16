# RecSys16(사분의 오) Movie Recommendation
![image](./images/movie.png)

## 프로젝트 소개
### 개요
MovieLens 데이터의 일부(31360 users, 6807 items, 5154471 interactions)를 활용, 사용자의 영화 시청 이력 데이터를 바탕
으로 사용자가 다음에 시청할 영화와 시청했던 영화 중 일부를 예측했다.
유저의 영화에 대한 평가 여부(implicit feedback)와 영화(아이템)에 대한 정보(title, year, genre, director, writer)가 주어졌으
며, timestamp 기반의 sequential recommendation 시나리오에서 일부 item이 누락된(dropout) 상황을 가정하여 단순히 마지
막 item만을 예측하는 문제보다 복잡하고 실제와 비슷한 상황을 설정했다.

### 모델
#### 제공된 baseline 코드와 함께 RecBole 라이브러리 사용
제공된 baseline 코드와 [RecBole 라이브러리](https://github.com/RUCAIBox/RecBole)를 활용해 CF, Autoencoder기반의 모델과 S3Rec, GRU4Rec, Bert4Rec 등
sequential 모델을 포함해 총 37종류의 모델을 학습시키고, 결과를 제출하였다. (recall@10: 0.1600 이 최대)

### 모델 성능 정리

![image](./images/model-recall.png)
모델 성능을 비교 분석해보며 얻은 결론은 크게 2가지였다. 첫번째는 sequential model보다 general model이 전체적으로
좋은 성능을 보여준다는 점이었다. Sequential model은 유저가 이전에 본 영화 리스트를 바탕으로 다음에 볼 영화를 예측
하는 모델이다. 이러한 점 때문에 유저가 중간에 본 영화에 대해서 예측을 하기 어려웠고, 결과적으로 general model의
추천 결과가 더 좋은 성능 점수를 보여주었다고 분석했다. 두번째는 딥러닝 기반이 아닌 모델이 좋은 성능을 보여준다는
점이었다. 단일 모델로 제일 좋은 성능을 보여준 모델은 hidden layer도 없는 linear model인 EASE(Embarrassingly
Shallow Autoencoders)였다. 이외에도 EASE를 발전시킨 ADMM-SLIM과 SLIMElastic도 꽤 좋은 성능(recall@10: 0.1300)을
보여주었다. 

### 앙상블 (public recall@10: 0.1600 -> 0.1878)

앙상블 방법은 Voting방식을 사용하였으며, 개별 성능을 확인했던 모델을 추가 및 삭제해가며 각 모델이 앙상블의 성능에 미치는 영향을 확인하였다. 
그 결과 성능이 좋은 General model만 사용하는 것이 아니라, Sequential model을 함께 앙상블 하였을 때 가장 좋은 성능을 보였다.
그 다음으로는, Top-K 반영 개수를 각각 15와 20으로 변경해 보았다. 이를 통해 Top11-20에도 정답이 분포하는 경우를 반
영하고자 했다. Top1-10을 반영하는 모델과 Top11-15를 반영하는 모델을 모두 사용했을 때 더 좋은 결과가 나타나는 것을 확인하였다.
이 결과를 바탕으로, Top1-10과 Top11-15에서 나오는 결과의 가중치를 조정하는 실험을 진행하였다. Top1-10과 Top11-15
의 가중치를 1:1부터 1:0.1까지 줄여서 실험하였으며, 그 결과 1:0.3의 비율로 앙상블 했을 때 가장 좋은 성능을 나타냈다.

## 팀원
| [ ![구창회](https://avatars.githubusercontent.com/u/63918561?v=4) ](https://github.com/sonyak-ku) | [ ![김지원](https://avatars.githubusercontent.com/u/97625330?v=4) ](https://github.com/Jiwon1729) | [ ![전민규](https://avatars.githubusercontent.com/u/85151359?v=4) ](https://github.com/alsrb0607) | [ ![정준우](https://avatars.githubusercontent.com/u/39089969?v=4) ](https://github.com/ler0n) |
|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|                             [ 구창회 ](https://github.com/sonyak-ku)                              |                             [ 김지원 ](https://github.com/Jiwon1729)                              |                              [ 전민규 ](https://github.com/alsrb0607)                             |                              [ 정준우 ](https://github.com/ler0n)                             |
|                              RankFM 구현, 모델 앙상블                             |                     BPR 구현, 모델 앙상블                    |                               S3Rec 개선, WandB&Sweep 세팅, 1~2주차 PM(WBS)                              |                        Recbole 라이브러리 inference 제작, 3~4주차 PM          |


## 최종 순위 및 결과

|리더보드| Recall@10 |   순위   |
|:--------:|:---------:|:------:|
|public|  0.1878   | **1위** |
|private|  0.1735   | **1위** |

![image](./images/private-movie.png)

## 참고자료
- [Wrap-up report & 발표자료](https://www.notion.so/343a3d95a9024967ae56061b697ac233)
