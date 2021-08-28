## CTAR Dataset and Generation Codes

This repository contains Causal Tags and Ratings (CTAR) dataset and generation codes.

If you have any questions, please feel free to contact us.

### Overview

Accurate recommendation and reliable explanation are two key issues for modern recommender systems. However, most recommendation benchmarks only concern on the prediction of user-item ratings while omit the underlying causes behind the ratings. For example, the widely-used Yahoo!R3 dataset contains little information on the causes of the user-movie ratings. A solution could be to conduct surveys and require the users to provide such information. In practice, the user surveys can hardly avoid compliance issues and  sparse user responses, which greatly hinders the exploration of causality based recommendation. To better support the studies of causal inference and further explanations in recommender systems, we propose to construct a semi-synthetic dataset with Causal Tags And Ratings (CTAR). The dataset is based on the movies as well as their descriptive tags and rating information collected from a famous movie rating website. Then, a causal graph is constructed to describe the causal mechanism of the practical movie recommendation scenarios, with the hyper-parameters determined by the collected data. Based on the collected data and the causal graph, the user-item-ratings and their corresponding user-item-tags are automatically generated, which provides the reasons (selected tags) why the user rates the items. The proposed data generation framework is not limited to recommendation, and the released APIs can be used to generate customized datasets for other research tasks.

### Requirements

- pandas
- numpy

### Get Started

##### Quick Start Example:

```bash
bash run.sh
```

##### Generate your own dataset:

We have set default hyperparameters in the dataset generation implementation. So the parameter arguments are optional for running the code. You can see parameter arguments in `generate_data.py` as follows:

```python
[--max_movie_num MAX_MOVIE_NUM] Maximum number of movies (Upper bound=9715)
[--max_user_num MAX_USER_NUM] Maximum number of users
[--max_user_like_tag MAX_USER_LIKE_TAG] Maximum number of users’ preferred tags
[--min_user_like_tag MIN_USER_LIKE_TAG] Minimum number of users’ preferred tags
[--max_tag_per_movie MAX_TAG_PER_MOVIE] Maximum number of movies’ tags
[--min_tag_per_movie MIN_TAG_PER_MOVIE] Minimum number of movies’ tags
[--rater RATER] Generation mechanism of rating
[--recsys RECSYS] Bias from recommendation system
[--rcttag_user_num RCTTAG_USER_NUM] Number of users in RCTTag
[--rcttag_movie_num RCTTAG_MOVIE_NUM] Number of movies in RCTTag
[--missing_rate_rating MISSING_RATE_RATING] Missing rate of rating
[--missing_type_rating MISSING_TYPE_RATING] Missing type of rating
[--missing_rate_obstag MISSING_RATE_OBSTAG] Missing rate of ObsTag
[--missing_type_obstag MISSING_TYPE_OBSTAG] Missing type of ObsTag
[--quality_sigma QUALITY_SIGMA] Standard deviation of normal distribution for quality
[--test_identifiable_num TEST_IDENTIFIABLE_NUM] Number of Test Dataset II
[--test_identifiable_num_positive TEST_IDENTIFIABLE_NUM_POSITIVE] Number of positive samples in Test Dataset II
[--test_inidentifiable_num TEST_INIDENTIFIABLE_NUM] Number of Test Dataset III
[--test_inidentifiable_positive TEST_INIDENTIFIABLE_POSITIVE] Number of positive samples in Test Dataset III
[--obstag_non_missing_rate OBSTAG_NON_MISSING_RATE] Probability of no missing in ObsTag
[--need_trainset NEED_TRAINSET] Whether need to generate train set
[--need_testset NEED_TESTSET] Whether need to generate test set
[--rerank_id RERANK_ID] Whether need to rerank it for dataset
```

### Dataset Description

There are four datasets for training: Movie, Rating, ObsTag and RCTTag. The summary of the dataset is showed in the table as follows.
|            | Filename       | Size  | Records | Data in each record   |
| :--------- | :------------- | :---- | :------ | :-------------------- |
| train      | movie.csv      | 34KB  | 1,000   | movieid,taglist       |
|            | rating.csv     | 192KB | 20,128  | userid,movieid,rating |
|            | obstag.csv     | 95KB  | 9,196   | userid,movieid,tagid  |
|            | rcttag.csv     | 16KB  | 1,529   | userid,movieid,tagid  |
| test       | test\_1\.csv   | 13KB  | 1,528   | userid,tagid,islike   |
|            | test\_2\.csv   | 46KB  | 5,000   | userid,tagid,islike   |
|            | test\_3\.csv   | 38KB  | 4,000   | userid,tagid,islike   |
|            | test.csv       | 97KB  | 10,528  | userid,tagid,islike   |
+ Movie:
This dataset contains the basic information. The field “taglist” contains the 8 tags for each movie. We assume that any tags in other datasets will appear in this file.
+ Rating:
This dataset contains the observed ratings for some user-movie pairs and it may suffer from the common biases existing in real world data. Rating for a particular user-movie pair mainly comes from two parts: the movie’s own speciality and the user’s preference to that movie. For the first part, each movie has its own speciality, thus we can treat this specialty as heterogeneity among movies. For the second part, in this problem, for a particular user-movie pair, we assume that the preference only comes from the number of tags of the movie that the user likes. That is, users tend to rate higher if a movie contains tags that he/she likes. The more tags a user likes, the higher the rating may be.
+ ObsTag:
This dataset contains the observed tags that users labeled to movies. In this problem, you may assume that users may only label tags that they like. If the field “tag” is labeled “-1”, you may assume that user does not like any tag of the corresponding movie. But when labelling a movie, a user may label fewer tags than what he/she really likes. For example, for a user-movie pair, if the user labeled “love”, it means that “love” is one of the 8 tags of that movie and the user really likes the tag “love”, but at the same time, the user may also like the other 7 tags and he/she may simply forget to label those tags by chance.Note that we have used tagid (e.g.,'1') to replace tags (e.g., 'love').
+ RCTTag:
This dataset is constructed from a random experiment. The users and the movies labeled by users are randomly selected. Also, users are forced to label all the tags they like for a particular movie. That is, if “-1” appears, you may still assume that the user does not like any tag of the movie. Besides, for tags of a movie that do not appear in this dataset for a user-movie pair, we are sure that the user does not like those tags.

### Intended Use and Future Research Tasks
Besides the tasks of predicting the rating of a user-movie pair or estimating a user's preference to a particular tag, e.g., [PCIC2021 Competitions Track2](https://competition.huaweicloud.com/information/1000041488/introduction) , the released CTAR dataset allows various other research directions, some of which are listed below:

+ Debiasing in recommender systems  
The CTAR dataset is generated using causal graphical model that simulates common biases and missingness mechanisms to make it as close to practical scenarios as possible.  All the counterfactuals in CTAR are available and can be used for evaluating novel causality-based debiasing methods.
+ Explainable recommendations  
Most recommender systems aim to rank movies in a descending order according to predicted ratings. The proposed CTAR dataset provides both user-movie ratings, and user-movie tags, and can be used for explainable recommendation research, e.g., to discover the causal mechanisms behind users’ preferences.
+ Counterfactual evaluation  
Traditional development and iteration of recommender models rely on large-scale online A/B tests, which are generally expensive,  time-consuming, and even unethical in some cases. Counterfactual evaluation has recently become a promising alternative approach as it allows offline evaluations of the online metrics, leading to a substantial increase in experimentation agility. In our semi-synthetic CTAR  dataset, both factual and counterfactual outcomes are known, allowing it to serve as a benchmark for counterfactual evaluation methods. 

### Documentation

You can see more details in our paper and support material.

### Citation

If you reference or use our methodology, code or results in your work, please consider citing:

