import numpy as np
import pandas as pd


def extract_data():
    bigtag = pd.read_csv('./data/train/bigtag.csv').values
    choicetag = pd.read_csv('./data/train/choicetag.csv').values
    movie_data = pd.read_csv('./data/train/movie.csv')
    movie = movie_data['taglist'].apply(eval).tolist()

    user_num = np.max(bigtag[:,0])
    tag_num = np.max(movie)

    mat = np.zeros((user_num+1,tag_num+1))
    all_data_array = []
    bigtag_array = []
    choicetag_array = []

    # extract deterministic data from bigtag
    for i in range(bigtag.shape[0]):
        if bigtag[i][2] != -1:
            mat[bigtag[i][0]][bigtag[i][2]] = 1
            all_data_array.append([bigtag[i][0],bigtag[i][2],1])
            bigtag_array.append([bigtag[i][0],bigtag[i][2],1])
        if bigtag[i][2] == -1:
            for tag in movie[bigtag[i][1]]:
                mat[bigtag[i][0]][tag] = -1
                all_data_array.append([bigtag[i][0],tag,0])
                bigtag_array.append([bigtag[i][0],tag,0])

    # extract deterministic data from choicetag
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            mat[choicetag[i][0]][choicetag[i][2]] = 1
            all_data_array.append([choicetag[i][0],choicetag[i][2],1])
            choicetag_array.append([choicetag[i][0],choicetag[i][2],1])
        if choicetag[i][2] == -1:
            for tag in movie[choicetag[i][1]]:
                mat[choicetag[i][0]][tag] = -1
                all_data_array.append([choicetag[i][0],tag,0])
                choicetag_array.append([choicetag[i][0],tag,0])
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            for tag in movie[choicetag[i][1]]:
                if mat[choicetag[i][0]][tag] == 0:
                    mat[choicetag[i][0]][tag] = -1
                    all_data_array.append([choicetag[i][0],tag,0])
                    choicetag_array.append([choicetag[i][0],tag,0])

    # Unique
    print("=======all data=======")
    all_data_array = np.array(all_data_array)
    print("---before unique---")
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))
    all_data_array = [tuple(row) for row in all_data_array]
    all_data_array = np.unique(all_data_array, axis=0)
    print("---after unique---")
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))

    # Unique
    print("=======bigtag=======")
    bigtag_array = np.array(bigtag_array)
    print("---before unique---")
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))
    bigtag_array = [tuple(row) for row in bigtag_array]
    bigtag_array = np.unique(bigtag_array, axis=0)
    print("---after unique---")
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))

    # Unique
    print("=======choicetag=======")
    choicetag_array = np.array(choicetag_array)
    print("---before unique---")
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))
    choicetag_array = [tuple(row) for row in choicetag_array]
    choicetag_array = np.unique(choicetag_array, axis=0)
    print("---after unique---")
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))

    bigtag_df = pd.DataFrame(bigtag_array, columns = ['userid','tagid','islike'])
    bigtag_df.to_csv("./data/train/extract_bigtag.csv", header=True, index=False)
    choicetag_df = pd.DataFrame(choicetag_array, columns = ['userid','tagid','islike'])
    choicetag_df.to_csv("./data/train/extract_choicetag.csv", header=True, index=False)
    all_data_df = pd.DataFrame(all_data_array, columns = ['userid','tagid','islike'])
    all_data_df.to_csv("./data/train/extract_alldata.csv", header=True, index=False)

extract_data()