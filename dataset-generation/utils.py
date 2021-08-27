import numpy as np
import pandas as pd


def random_tag(obstag_count: pd.DataFrame, size, rct_distribution):
    """
    从大标签中根据给定的权重生成一个标签列表
    generate a label list according to the weight from obstag_count
    """
    return np.random.choice(obstag_count.index,
                            size=size,
                            replace=False,
                            p=rct_distribution)


def generate_user_like_tag(user_id, user_tag_count, obstag_count: pd.DataFrame,
                           rct_distribution) -> pd.DataFrame:
    """
    对每个用户生成其喜欢的标签
    generate favorite tags for each user
    """
    user_like_tag = pd.DataFrame(
        columns=['userid', 'user_tag_count', 'user_like_tag'])
    for i, uid in enumerate(user_id):
        user_tag_list = random_tag(obstag_count, user_tag_count[i],
                                   rct_distribution)
        user_like_tag = user_like_tag.append(
            {
                'userid': uid,
                'user_tag_count': user_tag_count[i],
                'user_like_tag':
                obstag_count.loc[user_tag_list].index.tolist()
            },
            ignore_index=True)
    return user_like_tag.set_index('userid')


def get_rating(rating: pd.DataFrame,
               user_id,
               movie_id,
               user_like_tag_list,
               movie_real_tag_list,
               max_user_num,
               max_movie_num,
               rater='weightedsum',
               quality: pd.DataFrame = None):
    if rater == 'weightedsum':
        base_value = 10
        for uid, mid in [(x, y) for x in user_id for y in movie_id]:
            user_tag = user_like_tag_list.loc[uid, 'user_like_tag']
            movie_tag = movie_real_tag_list.loc[mid, 'taglist']
            score = 0
            for upos, tag in enumerate(user_tag):
                if tag in movie_tag:
                    mpos = movie_tag.index(tag)
                    score += base_value / (upos + 1) + base_value / (mpos + 1)
            rating.loc[uid, mid] = score
        rating = rating / max(rating) * 4
        rating += np.random.normal(size=(max_user_num, max_movie_num))
        rating = rating.clip(lower=0, upper=4).astype(int) + 1
    elif rater == 'qualitybase':
        for uid, mid in [(x, y) for x in user_id for y in movie_id]:
            user_tag = user_like_tag_list.loc[uid, 'user_like_tag']
            movie_tag = movie_real_tag_list.loc[mid, 'taglist']
            tag_scroe = len(set(user_tag).intersection(set(movie_tag)))
            quality_score = quality.loc[mid, 'quality']
            rating.loc[uid, mid] = quality_score-0.7 + \
                np.random.normal(loc=tag_scroe/2, scale=0.25, size=1)[0]
        rating = rating.applymap(round).clip(lower=1, upper=5)
    else:
        raise NotImplementedError

    return rating


def get_recsys_score(recsys: pd.DataFrame,
                     max_movie_num,
                     mv_tag_count: pd.Series,
                     rec_type='Pop') -> pd.DataFrame:
    if rec_type == 'Pop':
        recsys.loc[:, :] = mv_tag_count.head(
            max_movie_num).values[:, None].T  # 给每一个user进行赋值
    else:
        raise NotImplementedError
    return recsys


def get_r_rating(r_rating: pd.DataFrame,
                 missing_rate,
                 missing_type='default',
                 recsys=None) -> pd.DataFrame:
    user_id = r_rating.index
    movie_id = r_rating.columns
    if missing_type == 'default':
        # 默认算法，先用logit函数根据rank生成默认概率，再乘以缺失率
        logit_slope = 0.002
        logit_intersection = -len(movie_id) / 2 * logit_slope
        p = np.array(range(len(movie_id), 0, -1)) * \
            logit_slope+logit_intersection
        p = 1 / (1 + np.exp(-p))
        P = (p / p.mean() * missing_rate).reshape((1, len(movie_id)))
        r_rating.loc[:, :] = np.random.binomial(1,
                                                p,
                                                size=(len(user_id),
                                                      len(movie_id)))
    else:
        return NotImplementedError
    return r_rating


def get_r_obstag(r_obstag: pd.DataFrame,
                 missing_rate,
                 missing_type='default',
                 recsys: pd.DataFrame = None,
                 rating: pd.DataFrame = None) -> pd.DataFrame:
    user_id = r_obstag.index
    movie_id = r_obstag.columns
    logit_slope = 0.005
    logit_intersection = -len(movie_id) * logit_slope
    if missing_type == 'default':
        # 默认user_specific与system_bias同权重
        for uid in user_id:
            # 与movie_rank统一scale
            user_rating = np.array(rating.loc[uid]) / 5 * len(movie_id)
            p = np.array(range(len(movie_id), 0, -1)) + user_rating
            p = p * logit_slope + logit_intersection
            p = 1 / (1 + np.exp(-p))
            p = (p / p.mean() * missing_rate).reshape((1, len(movie_id)))
            r_obstag.loc[uid, :] = np.random.binomial(1, p)
    else:
        raise NotImplementedError
    return r_obstag


def split_test_val(data_pd: pd.DataFrame, val_num):
    val_num = int(val_num)
    data_index = np.array(data_pd.index)
    np.random.shuffle(data_index)
    total_num = len(data_index)
    val_set = data_pd.loc[data_index[0:val_num]]
    test_set = data_pd.loc[data_index[val_num:total_num]]
    return val_set, test_set


def check_test_set(test_set: pd.DataFrame, user_like_tag_list: pd.DataFrame):
    wrongnum = 0
    for i in test_set.index:
        uid = test_set.loc[i]['userid']
        tag = test_set.loc[i]['tagid']
        like = test_set.loc[i]['like']
        userlike = user_like_tag_list.loc[uid]['user_like_tag']
        if tag in userlike and like == 0:
            wrongnum += 1
        elif tag not in userlike and like == 1:
            wrongnum += 1
    return wrongnum


def rerankid_test_data(filepath, filename, tag_dict, savepath):
    test_df = pd.read_csv(filepath + filename)
    test_df['tagid'] = test_df['tagid'].apply(lambda x: tag_dict[x])
    test_df.to_csv(savepath + filename , header=True, index=False)

