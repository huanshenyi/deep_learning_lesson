import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np


"""
花の認識
"""
import ai_utils

DATA_FILE = './data/Iris.csv'

SPECIES = [
    'Iris-setosa', #ヒオウギアヤメ
    'Iris-versicolor',#变色
    'Iris-virginica' #
]

#花の特徴     #萼(がく)長さ      #萼(がく)の横幅   #花ビラ長さ      #花びら横幅　
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def get_pred_label(test_sample_feat,train_data):
    """

    似た者同士でラベル取得

    """
    dis_list = []

    for idx, row in train_data.iterrows():
        #ライニングサンプル特徴を取得
        train_sample_feat = row[FEAT_COLS].values

        #距離を計算
        dis = euclidean(test_sample_feat, train_sample_feat)
        dis_list.append(dis)

    # 最短距離の位置情報
    pos = np.argmin(dis_list)
    pred_label = train_data.iloc[pos]['Species']
    return pred_label

def main():

    #pandas ファイル読み込み
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')

    #EDA
    ai_utils.do_eda_plot_for_iris(iris_data)

    #データ集落
    train_data, test_data = train_test_split(iris_data, test_size=1/3, random_state=10)

    #予測正解率
    acc_count = 0

    #アルゴリズム
    for idx, row in test_data.iterrows():
        #サンプル特徴
        test_sample_feat = row[FEAT_COLS].values

        #予想値
        pred_label = get_pred_label(test_sample_feat, train_data)

        #実際値
        true_label = row['Species']

        print('サンプル{}のラベルは{},予想ラベルは{}'.format(idx, true_label, pred_label))
        if true_label == pred_label:
            acc_count += 1
    #正解率

    accuracy = acc_count / test_data.shape[0]
    print('正解率は{:.2f}%'.format(accuracy * 100))

if __name__ =='__main__':
    main()
