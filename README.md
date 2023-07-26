# POLIMI Recommender Systems Challenge 2018
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

The objective of the competition was to create the best recommeder system for a streaming service by providing 10 recommended items. The evaluation metric was MAP@10.

In this repo we report the various experiments made and the various evolutions of our Recommender System.

[Link to the official website of the challenge](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi)

## Results

The end result was a hybrid Recommender System formed by RP3betaRecommender and two versions of SLIMElasticNetRecommender specialized for different portions of users. ([See last submission](Daniele/Recommenders/LastDance/sub_hybrid.py))

We used both an implicit and an explicit matrix, normalizing the values using a dynamic Logistic function according to the item bias and user bias, see the implementation [here](Daniele/Utils/MatrixManipulation.py) of the funciton explicitURM.

* MAP@10 = 0.06021 on Kaggle's public leaderboard

## Team
* [Daniele Cipollone](https://github.com/DanCip00)
* [Federico Bono](https://github.com/FredBonux)
