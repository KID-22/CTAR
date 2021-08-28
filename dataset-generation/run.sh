echo "begin"
python generate_data.py \
--max_movie_num 1000 \
--max_user_num 1000 \
--max_user_like_tag 20 \
--min_user_like_tag 10 \
--max_tag_per_movie 8 \
--min_tag_per_movie 8 \
--rater qualitybase \
--recsys Pop \
--rcttag_user_num 100 \
--rcttag_movie_num 10 \
--missing_rate_rating 0.02 \
--missing_type_rating default \
--missing_rate_obstag 0.007 \
--missing_type_obstag default \
--quality_sigma 0.75 \
--test_identifiable_num 5000 \
--test_identifiable_num_positive 1500 \
--test_inidentifiable_num 4000 \
--test_inidentifiable_positive 1200 \
--obstag_non_missing_rate 0.6 \
--need_trainset 1 \
--need_testset 1 \
--rerank_id 1
echo "end"
