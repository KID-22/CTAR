import os
import sys
import pandas as pd
import numpy as np
import random
from utils import *

np.random.seed(9527)
random.seed(9527)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_movie_num', type=int, default=1000)
    parser.add_argument('--max_user_num', type=int, default=1000)
    parser.add_argument('--max_user_like_tag', type=int, default=20)
    parser.add_argument('--min_user_like_tag', type=int, default=10)
    parser.add_argument('--max_tag_per_movie', type=int, default=8)
    parser.add_argument('--min_tag_per_movie', type=int, default=8)
    parser.add_argument('--rater', type=str, default='qualitybase')
    parser.add_argument('--recsys', type=str, default='Pop')
    parser.add_argument('--rcttag_user_num', type=int, default=100)
    parser.add_argument('--rcttag_movie_num', type=int, default=10)
    parser.add_argument('--missing_rate_rating', type=float, default=0.02)
    parser.add_argument('--missing_type_rating', type=str, default='default')
    parser.add_argument('--missing_rate_obstag', type=float, default=0.007)
    parser.add_argument('--missing_type_obstag', type=str, default='default')
    parser.add_argument('--quality_sigma', type=float, default=0.75)
    parser.add_argument('--test_identifiable_num', type=int, default=5000)
    parser.add_argument('--test_identifiable_num_positive',
                        type=int,
                        default=1500)
    parser.add_argument('--test_inidentifiable_num', type=int, default=4000)
    parser.add_argument('--test_inidentifiable_positive',
                        type=int,
                        default=1200)
    parser.add_argument('--obstag_non_missing_rate', type=float, default=0.6)
    parser.add_argument('--need_trainset', type=int, default=0)
    parser.add_argument('--need_testset', type=int, default=0)
    parser.add_argument('--rerank_id', type=int, default=1)

    args = parser.parse_args()
    paras = vars(args)

    data_dir = './'
    if not os.path.exists(data_dir + 'generate_data/'):
        os.makedirs(data_dir + 'generate_data/')
    for i in ['train', 'test']:
        if not os.path.exists(data_dir + 'final_data/before_rerank_id/' + i):
            os.makedirs(data_dir + 'final_data/before_rerank_id/' + i)
        if not os.path.exists(data_dir + 'final_data/rerank_id/' + i):
            os.makedirs(data_dir + 'final_data/rerank_id/' + i)

    big_movie_tag_ct = pd.read_csv(data_dir + 'original_data/movie_tag_ct.csv')
    base_movie_rating = pd.read_csv(data_dir +
                                    'original_data/movie_rating.csv',
                                    index_col='movieid')

    print('======generating base data======')

    # generate user_id data
    if os.path.exists(data_dir + 'generate_data/user_id.csv'):
        user_id = np.array(
            pd.read_csv(data_dir + 'generate_data/user_id.csv',
                        index_col='userid').index)
        max_user_num = len(user_id)
    else:
        max_user_num = paras['max_user_num']
        user_id = np.array(range(max_user_num))
        pd.DataFrame(data=user_id,
                     columns=['userid']).set_index('userid').to_csv(
                         data_dir + 'generate_data/user_id.csv', header=True)

    # generate movie_id data
    mv_tag_count: pd.DataFrame = big_movie_tag_ct[[
        'movieid', 'tagCount'
    ]].groupby('movieid')['tagCount'].sum().sort_values(
        ascending=False)  # 每部电影被多少人次打过tag
    if os.path.exists(data_dir + 'generate_data/movie_id.csv'):
        movie_data = pd.read_csv(data_dir + 'generate_data/movie_id.csv',
                                 index_col='movieid')
        movie_id = np.array(movie_data.index)
        max_movie_num = len(movie_id)
    else:
        max_movie_num = min(len(mv_tag_count), paras['max_movie_num'])
        movie_id = np.array(mv_tag_count.head(max_movie_num).index)
        movie_data = pd.DataFrame(data=movie_id,
                                  columns=['movieid']).set_index('movieid')
        movie_data.to_csv(data_dir + 'generate_data/movie_id.csv', header=True)

    # generate obstag_count data
    if os.path.exists(data_dir + 'generate_data/obstag_count.csv'):
        obstag_count = pd.read_csv(
            data_dir + 'generate_data/obstag_count.csv',
            index_col='tagid')
        rct_distribution = obstag_count['tagCount'] / obstag_count['tagCount'].sum()  # 生成标签流行度的分布
    else:
        obstag_count: pd.DataFrame = big_movie_tag_ct[
            big_movie_tag_ct['movieid'].isin(movie_id)].groupby(
                'tagid')['tagCount'].sum().sort_values(
                    ascending=False).to_frame()  # 统计选出来的电影集合里，以标签为单位，每个标签的总数量
        obstag_count.to_csv(data_dir + 'generate_data/obstag_count.csv', header=True)
        rct_distribution = obstag_count['tagCount'] / obstag_count['tagCount'].sum()  # 生成标签流行度的分布

    # generate movie_real_tag_list data
    if os.path.exists(data_dir + 'generate_data/movie_real_tag_list.csv'):
        movie_real_tag_list = pd.read_csv(
            data_dir + 'generate_data/movie_real_tag_list.csv',
            index_col='movieid')
        movie_real_tag_list['taglist'] = movie_real_tag_list['taglist'].apply(
            eval)
        # print(movie_real_tag_list.head())
    else:
        movie_real_tag_list = pd.DataFrame()
        for mid in movie_id:
            mv_tag = big_movie_tag_ct[big_movie_tag_ct['movieid'] ==
                                      mid]['tagid'].to_list()
            mv_tag = mv_tag[:paras['max_tag_per_movie']]  # 如果一部电影标签过多，删除过多的标签
            while len(mv_tag) < paras['min_tag_per_movie']:  # 如果一部电影标签过少，补全标签
                newtag = random_tag(obstag_count, 1, rct_distribution)[0]
                if newtag not in mv_tag:
                    mv_tag.append(newtag)
            movie_real_tag_list = movie_real_tag_list.append(
                {
                    'movieid': mid,
                    'taglist': mv_tag
                }, ignore_index=True)
        movie_real_tag_list['movieid'] = movie_real_tag_list['movieid'].astype(
            'int64')
        movie_real_tag_list = movie_real_tag_list.set_index('movieid')
        movie_real_tag_list.to_csv(data_dir +
                                   'generate_data/movie_real_tag_list.csv',
                                   header=True)

    # generate user_like_tag_list data
    if os.path.exists(data_dir + 'generate_data/user_like_tag_list.csv'):
        user_like_tag_list = pd.read_csv(
            data_dir + 'generate_data/user_like_tag_list.csv',
            index_col='userid')
        user_like_tag_list['user_like_tag'] = user_like_tag_list[
            'user_like_tag'].apply(eval)
        # print(user_like_tag_list.head())
    else:
        max_user_like_tag = paras['max_user_like_tag']
        min_user_like_tag = paras['min_user_like_tag']
        user_tag_count = np.random.randint(
            low=min_user_like_tag,
            high=max_user_like_tag + 1,
            size=max_user_num)  # 生成每个user喜欢的标签数量

        user_like_tag_list: pd.DataFrame = generate_user_like_tag(
            user_id, user_tag_count, obstag_count,
            rct_distribution)  # 生成每个用户喜欢的标签
        user_like_tag_list.to_csv(data_dir +
                                  'generate_data/user_like_tag_list.csv',
                                  header=True)

    # generate Real_RCTtag
    if os.path.exists(data_dir + 'generate_data/real_rcttag.csv'):
        real_rcttag = pd.read_csv(data_dir + 'generate_data/real_rcttag.csv',
                                  index_col='userid')
        real_rcttag.columns = map(eval, real_rcttag.columns)
        real_rcttag = real_rcttag.applymap(eval)
        # print(real_rcttag.head())
    else:
        real_rcttag: pd.DataFrame = pd.DataFrame(index=user_id,
                                                 columns=movie_id)
        real_rcttag.index.name = 'userid'
        for uid, mid in [(x, y) for x in user_id for y in movie_id]:
            mv_tag = movie_real_tag_list.loc[mid, 'taglist']
            user_tag = user_like_tag_list.loc[uid, 'user_like_tag']
            real_rcttag.loc[uid, mid] = list(
                set(mv_tag).intersection(set(user_tag)))
        real_rcttag.to_csv(data_dir + 'generate_data/real_rcttag.csv',
                           header=True)

    # generate Quality data
    if os.path.exists(data_dir + 'generate_data/quality.csv'):
        quality = pd.read_csv(data_dir + 'generate_data/quality.csv',
                              index_col='movieid')
        # print(quality.head())
    else:
        quality_sigma = paras['quality_sigma']
        quality: pd.DataFrame = pd.DataFrame(index=movie_id,
                                             columns=['quality'])
        quality.index.name = 'movieid'
        quality['quality'] = base_movie_rating.loc[movie_id] + \
            np.random.normal(loc=0, scale=quality_sigma,
                             size=len(movie_id)).reshape(-1, 1)
        quality.to_csv(data_dir + 'generate_data/quality.csv', header=True)

    # generate Rating data
    if os.path.exists(data_dir + 'generate_data/rating.csv'):
        rating = pd.read_csv(data_dir + 'generate_data/rating.csv',
                             index_col='userid')
        rating.columns = map(eval, rating.columns)
        # print(rating.head())
    else:
        rating: pd.DataFrame = pd.DataFrame(index=user_id, columns=movie_id)
        rating.index.name = 'userid'
        rater = paras['rater']
        rating = get_rating(rating,
                            user_id=user_id,
                            movie_id=movie_id,
                            user_like_tag_list=user_like_tag_list,
                            movie_real_tag_list=movie_real_tag_list,
                            max_user_num=max_user_num,
                            max_movie_num=max_movie_num,
                            rater=rater,
                            quality=quality)
        rating.to_csv(data_dir + 'generate_data/rating.csv', header=True)

    # generate Recsys
    if os.path.exists(data_dir + 'generate_data/recsys.csv'):
        recsys = pd.read_csv(data_dir + 'generate_data/recsys.csv',
                             index_col='userid')
        recsys.columns = map(eval, recsys.columns)
        # print(recsys.head())
    else:
        recsys: pd.DataFrame = pd.DataFrame(index=user_id, columns=movie_id)
        recsys.index.name = 'userid'
        recsys = get_recsys_score(recsys,
                                  max_movie_num=max_movie_num,
                                  mv_tag_count=mv_tag_count)
        recsys.to_csv(data_dir + 'generate_data/recsys.csv', header=True)

    # generate R_RCTTag
    if os.path.exists(data_dir + 'generate_data/r_rcttag.csv'):
        r_rcttag = pd.read_csv(data_dir + 'generate_data/r_rcttag.csv',
                               index_col='userid')
        r_rcttag.columns = map(eval, r_rcttag.columns)
    else:
        rcttag_user_num = paras['rcttag_user_num']
        rcttag_movie_num = paras['rcttag_movie_num']
        r_rcttag: pd.DataFrame = pd.DataFrame(data=0,
                                              index=user_id,
                                              columns=movie_id)
        r_rcttag.index.name = 'userid'
        u_list = np.random.choice(user_id, size=rcttag_user_num, replace=False)
        for uid in u_list:
            m_list = np.random.choice(movie_id,
                                      size=rcttag_movie_num,
                                      replace=False)
            r_rcttag.loc[uid, m_list] = 1
        r_rcttag.to_csv(data_dir + 'generate_data/r_rcttag.csv', header=True)

    # generate R-Rating
    if os.path.exists(data_dir + 'generate_data/r_rating.csv'):
        r_rating = pd.read_csv(data_dir + 'generate_data/r_rating.csv',
                               index_col='userid')
        r_rating.columns = map(eval, r_rating.columns)
    else:
        missing_rate = paras['missing_rate_rating']
        missing_type = paras['missing_type_rating']
        r_rating: pd.DataFrame = pd.DataFrame(data=0,
                                              index=user_id,
                                              columns=movie_id)
        r_rating.index.name = 'userid'
        r_rating = get_r_rating(r_rating,
                                missing_rate=missing_rate,
                                missing_type=missing_type,
                                recsys=recsys)
        r_rating.to_csv(data_dir + 'generate_data/r_rating.csv', header=True)

    # generate R-Obstag
    if os.path.exists(data_dir + 'generate_data/r_obstag.csv'):
        r_obstag = pd.read_csv(data_dir + 'generate_data/r_obstag.csv',
                               index_col='userid')
        r_obstag.columns = map(eval, r_obstag.columns)
    else:
        missing_rate = paras['missing_rate_obstag']
        missing_type = paras['missing_type_obstag']
        r_obstag: pd.DataFrame = pd.DataFrame(data=0,
                                              index=user_id,
                                              columns=movie_id)
        r_obstag.index.name = 'userid'
        r_obstag = get_r_obstag(r_obstag,
                                missing_rate=missing_rate,
                                missing_type=missing_type,
                                recsys=recsys,
                                rating=rating)
        r_obstag.to_csv(data_dir + 'generate_data/r_obstag.csv', header=True)

    # generate train data
    print('======generating train data======')
    need_trainset = paras['need_trainset']
    if need_trainset == 1:
        # output movie data
        movie_data['taglist'] = movie_real_tag_list['taglist'].map(
            lambda xx: ','.join([str(x) for x in xx]))
        movie_data[['taglist']].to_csv(
            data_dir + 'final_data/before_rerank_id/train/movie.csv',
            header=True,
            index=True)
        # movie_data = pd.read_csv(data_dir + 'final_data/train/movie.csv')
        # print(movie_data.head())

        # output rating, rcttag and obstag
        rating_out = pd.DataFrame(columns=['userid', 'movieid', 'rating'])
        rcttag_out = pd.DataFrame(columns=['userid', 'movieid', 'tagid'])
        obstag_out = pd.DataFrame(columns=['userid', 'movieid', 'tagid'])
        obstag_missing = pd.DataFrame(columns=['userid', 'movieid', 'tagid'])
        obstag_nonmissing_rate = paras['obstag_non_missing_rate']

        missingcount = 0
        nonmissingcount = 0
        for uid, mid in [(x, y) for x in user_id for y in movie_id]:
            if r_rating.loc[uid, mid] == 1:
                rating_out = rating_out.append(
                    {
                        'userid': uid,
                        'movieid': mid,
                        'rating': rating.loc[uid, mid]
                    },
                    ignore_index=True)

            tmp_tag = real_rcttag.loc[uid, mid]
            if r_rcttag.loc[uid, mid] == 1:
                if len(tmp_tag) == 0:
                    rcttag_out = rcttag_out.append(
                        {
                            'userid': uid,
                            'movieid': mid,
                            'tagid': -1
                        },
                        ignore_index=True)
                for tag in tmp_tag:
                    rcttag_out = rcttag_out.append(
                        {
                            'userid': uid,
                            'movieid': mid,
                            'tagid': tag
                        },
                        ignore_index=True)
            if r_obstag.loc[uid, mid] == 1:
                if len(tmp_tag) == 0:
                    obstag_out = obstag_out.append(
                        {
                            'userid': uid,
                            'movieid': mid,
                            'tagid': -1
                        },
                        ignore_index=True)
                for i, tag in enumerate(tmp_tag):
                    if i == 0 or (i > 0 and
                                  np.random.random() < obstag_nonmissing_rate):
                        if i > 0:
                            missingcount += 1
                        obstag_out = obstag_out.append(
                            {
                                'userid': uid,
                                'movieid': mid,
                                'tagid': tag
                            },
                            ignore_index=True)
                    elif i > 0:
                        obstag_missing = obstag_missing.append(
                            {
                                'userid': uid,
                                'movieid': mid,
                                'tagid': tag
                            },
                            ignore_index=True)
                        nonmissingcount += 1
        rating_out['userid'] = rating_out['userid'].astype('int64')
        rating_out['movieid'] = rating_out['movieid'].astype('int64')
        rating_out['rating'] = rating_out['rating'].astype('int64')
        rating_out.to_csv(data_dir +
                          'final_data/before_rerank_id/train/rating.csv',
                          header=True,
                          index=False)
        rcttag_out['userid'] = rcttag_out['userid'].astype('int64')
        rcttag_out['movieid'] = rcttag_out['movieid'].astype('int64')
        rcttag_out.to_csv(data_dir +
                          'final_data/before_rerank_id/train/rcttag.csv',
                          header=True,
                          index=False)
        obstag_out['userid'] = obstag_out['userid'].astype('int64')
        obstag_out['movieid'] = obstag_out['movieid'].astype('int64')
        obstag_out.to_csv(data_dir +
                          'final_data/before_rerank_id/train/obstag.csv',
                          header=True,
                          index=False)
        obstag_missing['movieid'] = obstag_missing['movieid'].astype('int64')
        obstag_missing.to_csv(data_dir + 'generate_data/obstag_missing.csv',
                              header=True,
                              index=False)
        print("non mising rate",
              missingcount / (missingcount + nonmissingcount))
        print("non missing count", missingcount, "missingcount",
              nonmissingcount)

    print('======generating test set======')
    # generate test set
    need_testset = paras['need_testset']
    if need_testset == 1:
        test_identifiable_num = paras['test_identifiable_num']
        test_identifiable_num_positive = paras[
            'test_identifiable_num_positive']
        test_inidentifiable_num = paras['test_inidentifiable_num']
        test_inidentifiable_positive = paras['test_inidentifiable_positive']
        test_set: pd.DataFrame = pd.DataFrame(
            columns=['userid', 'tagid', 'islike'])

        if not os.path.exists(data_dir + 'generate_data/extract.csv'):
            os.system('python extract_data.py')
        extract_pd: pd.DataFrame = pd.read_csv(data_dir +
                                               'generate_data/extract.csv')
        extract_dict = dict(
            zip(zip(extract_pd['userid'], extract_pd['tagid']),
                extract_pd['islike']))

        # generating obstag missing data
        obstag_missing: pd.DataFrame = pd.read_csv(
            data_dir + 'generate_data/obstag_missing.csv')[['userid', 'tagid']]
        obstag_missing['islike'] = 1
        obstag_missing_dict = dict(
            zip(zip(obstag_missing['userid'], obstag_missing['tagid']),
                obstag_missing['islike']))

        test_1 = obstag_missing
        test_set = test_set.append(test_1, ignore_index=True)

        # generating identifiable data
        positive_count = 0
        negtive_count = 0
        rating_out = pd.read_csv(
            data_dir + 'final_data/before_rerank_id/train/rating.csv')
        tmp_pos = pd.DataFrame(columns=['userid', 'tagid', 'islike'])
        tmp_neg = pd.DataFrame(columns=['userid', 'tagid', 'islike'])
        for i in range(test_identifiable_num):
            find_tag = False
            while not find_tag:
                tmprating = rating_out.sample(n=1, axis=0)
                uid = int(tmprating['userid'])
                mid = int(tmprating['movieid'])
                for tmptag in movie_real_tag_list.loc[mid]['taglist']:
                    if not ((uid, tmptag) in obstag_missing_dict or
                            (uid, tmptag) in extract_dict):
                        if tmptag in user_like_tag_list.loc[uid][
                                'user_like_tag'] and positive_count < test_identifiable_num_positive:
                            tmp_pos = tmp_pos.append(
                                {
                                    'userid': uid,
                                    'tagid': tmptag,
                                    'islike': 1
                                },
                                ignore_index=True)
                            find_tag = True
                            positive_count += 1
                            break
                        elif tmptag not in user_like_tag_list.loc[uid][
                                'user_like_tag'] and negtive_count < test_identifiable_num - test_identifiable_num_positive:
                            tmp_neg = tmp_neg.append(
                                {
                                    'userid': uid,
                                    'tagid': tmptag,
                                    'islike': 0
                                },
                                ignore_index=True)
                            find_tag = True
                            negtive_count += 1
                            break

        test_2_pos = tmp_pos
        test_2 = tmp_neg
        test_2 = test_2.append(test_2_pos, ignore_index=True)
        test_set = test_set.append(test_2, ignore_index=True)
        test_2_dict = dict(
            zip(zip(test_2['userid'], test_2['tagid']), test_2['islike']))

        # generating inidentifiable data
        positive_count = 0
        negtive_count = 0
        rating_ut_dict = {}
        tmp_pos = pd.DataFrame(columns=['userid', 'tagid', 'islike'])
        tmp_neg = pd.DataFrame(columns=['userid', 'tagid', 'islike'])
        for i in rating_out.index:
            uid = int(rating_out.loc[i]['userid'])
            mid = int(rating_out.loc[i]['movieid'])
            for tmptag in movie_real_tag_list.loc[mid]['taglist']:
                rating_ut_dict[(uid, tmptag)] = 1
        for i in range(test_inidentifiable_num):
            find_tag = False
            while not find_tag:
                uid = np.random.choice(user_id)
                mid = np.random.choice(movie_id)
                if r_rating.loc[uid, mid] == 1:
                    continue
                for tmptag in movie_real_tag_list.loc[mid]['taglist']:
                    if not ((uid, tmptag) in rating_ut_dict or
                            (uid, tmptag) in obstag_missing_dict or
                            (uid, tmptag) in extract_dict or
                            (uid, tmptag) in test_2_dict):
                        if tmptag in user_like_tag_list.loc[uid][
                                'user_like_tag'] and positive_count < test_inidentifiable_positive:
                            tmp_pos = tmp_pos.append(
                                {
                                    'userid': uid,
                                    'tagid': tmptag,
                                    'islike': 1
                                },
                                ignore_index=True)
                            find_tag = True
                            positive_count += 1
                            break
                        elif tmptag not in user_like_tag_list.loc[uid][
                                'user_like_tag'] and negtive_count < test_inidentifiable_num - test_inidentifiable_positive:
                            tmp_neg = tmp_neg.append(
                                {
                                    'userid': uid,
                                    'tagid': tmptag,
                                    'islike': 0
                                },
                                ignore_index=True)
                            find_tag = True
                            negtive_count += 1
                            break
        test_3_pos = tmp_pos
        test_3 = tmp_neg
        test_3 = test_3.append(test_3_pos, ignore_index=True)
        test_set = test_set.append(test_3, ignore_index=True)

        test_set.to_csv(data_dir + 'final_data/before_rerank_id/test/test.csv',
                        header=True,
                        index=False)
        test_1.to_csv(data_dir + 'final_data/before_rerank_id/test/test_1.csv',
                      header=True,
                      index=False)
        test_2.to_csv(data_dir + 'final_data/before_rerank_id/test/test_2.csv',
                      header=True,
                      index=False)
        test_3.to_csv(data_dir + 'final_data/before_rerank_id/test/test_3.csv',
                      header=True,
                      index=False)

    # rerank movieid and tagid
    rerank_id = paras['rerank_id']
    if rerank_id == 1:
        movie_df = pd.read_csv(data_dir +
                               'final_data/before_rerank_id/train/movie.csv')
        # print(movie_df.head())
        movie_dict = {}
        tag_dict = {}
        tag_cnt = 0
        for index, row in movie_df.iterrows():
            movie_dict[row['movieid']] = index
            tag_list = row['taglist'].split(',')
            for tag in tag_list:
                if int(tag) not in tag_dict:
                    tag_dict[int(tag)] = tag_cnt
                    tag_cnt += 1
        tag_dict[-1] = -1
        print("number of tag: ", tag_cnt)

        movie_df['movieid'] = movie_df['movieid'].apply(
            lambda x: movie_dict[x])
        movie_df['taglist'] = movie_df['taglist'].apply(
            lambda x: ','.join([str(tag_dict[int(i)]) for i in x.split(',')]))
        movie_df.to_csv(data_dir + 'final_data/rerank_id/train/movie.csv',
                        header=True,
                        index=False)

        rating_df = pd.read_csv(data_dir +
                                'final_data/before_rerank_id/train/rating.csv')
        rating_df['movieid'] = rating_df['movieid'].apply(
            lambda x: movie_dict[x])
        rating_df.to_csv(data_dir + 'final_data/rerank_id/train/rating.csv',
                         header=True,
                         index=False)

        obstag_df = pd.read_csv(data_dir +
                                'final_data/before_rerank_id/train/obstag.csv')
        obstag_df['movieid'] = obstag_df['movieid'].apply(
            lambda x: movie_dict[x])
        obstag_df['tagid'] = obstag_df['tagid'].apply(lambda x: tag_dict[x])
        obstag_df.to_csv(data_dir + 'final_data/rerank_id/train/obstag.csv',
                         header=True,
                         index=False)

        rcttag_df = pd.read_csv(data_dir +
                                'final_data/before_rerank_id/train/rcttag.csv')
        rcttag_df['movieid'] = rcttag_df['movieid'].apply(
            lambda x: movie_dict[x])
        rcttag_df['tagid'] = rcttag_df['tagid'].apply(lambda x: tag_dict[x])
        rcttag_df.to_csv(data_dir + 'final_data/rerank_id/train/rcttag.csv',
                         header=True,
                         index=False)

        rerankid_test_data('final_data/before_rerank_id/test/', 'test.csv',
                           tag_dict, 'final_data/rerank_id/test/')
        rerankid_test_data('final_data/before_rerank_id/test/', 'test_1.csv',
                           tag_dict, 'final_data/rerank_id/test/')
        rerankid_test_data('final_data/before_rerank_id/test/', 'test_2.csv',
                           tag_dict, 'final_data/rerank_id/test/')
        rerankid_test_data('final_data/before_rerank_id/test/', 'test_3.csv',
                           tag_dict, 'final_data/rerank_id/test/')

    print('======Done!======')


if __name__ == "__main__":
    main()
