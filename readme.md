## Generation for CTAR Dataset

If you have any problems, please feel free to contact us.

Please cite our paper if this dataset helps your research.

### Overview



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
[--test_identifiable_num TEST_IDENTIFIABLE_NUM] Number of test Subset II
[--test_identifiable_num_positive TEST_IDENTIFIABLE_NUM_POSITIVE] Number of positive samples in test Subset II
[--test_inidentifiable_num TEST_INIDENTIFIABLE_NUM] Number of test Subset III
[--test_inidentifiable_positive TEST_INIDENTIFIABLE_POSITIVE] Number of positive samples in test Subset III
[--obstag_non_missing_rate OBSTAG_NON_MISSING_RATE] Probability of no missing in ObsTag
[--need_trainset NEED_TRAINSET] Whether need to generate train set
[--need_testset NEED_TESTSET] Whether need to generate test set
[--validation_percentage VALIDATION_PERCENTAGE] Proportion of validation set
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
| validation | valid\_1\.csv  | 4KB   | 458     | userid,movieid,islike |
|            | valid\_2\.csv  | 14KB  | 1,500   | userid,movieid,islike |
|            | valid\_3\.csv  | 12KB  | 1,200   | userid,movieid,islike |
|            | validation.csv | 30KB  | 3,158   | userid,movieid,islike |
| test       | test\_1\.csv   | 10KB  | 1,070   | userid,movieid,islike |
|            | test\_2\.csv   | 33KB  | 3,500   | userid,movieid,islike |
|            | test\_3\.csv   | 27KB  | 2,800   | userid,movieid,islike |
|            | test.csv       | 68KB  | 7,370   | userid,movieid,islike |
+ Movie:
This dataset contains the basic information. The field “taglist” contains the 8 tags for each movie. We assume that any tags in other datasets will appear in this file.
+ Rating:
This dataset contains the observed ratings for some user-movie pairs and it may suffer from the common biases existing in real world data. Rating for a particular user-movie pair mainly comes from two parts: the movie’s own speciality and the user’s preference to that movie. For the first part, each movie has its own speciality, thus we can treat this specialty as heterogeneity among movies. For the second part, in this problem, for a particular user-movie pair, we assume that the preference only comes from the number of tags of the movie that the user likes. That is, users tend to rate higher if a movie contains tags that he/she likes. The more tags a user likes, the higher the rating may be.
+ ObsTag:
This dataset contains the observed tags that users labeled to movies. In this problem, you may assume that users may only label tags that they like. If the field “tag” is labeled “-1”, you may assume that user does not like any tag of the corresponding movie. But when labelling a movie, a user may label fewer tags than what he/she really likes. For example, for a user-movie pair, if the user labeled “love”, it means that “love” is one of the 8 tags of that movie and the user really likes the tag “love”, but at the same time, the user may also like the other 7 tags and he/she may simply forget to label those tags by chance.Note that we have used tagid (e.g.,'1') to replace tags (e.g., 'love').
+ RCTTag:
This dataset is constructed from a random experiment. The users and the movies labeled by users are randomly selected. Also, users are forced to label all the tags they like for a particular movie. That is, if “-1” appears, you may still assume that the user does not like any tag of the movie. Besides, for tags of a movie that do not appear in this dataset for a user-movie pair, we are sure that the user does not like those tags.



### Documentation

You can see more details in our paper and support material.

### Citation

If you reference or use our methodology, code or results in your work, please consider citing:

### References

