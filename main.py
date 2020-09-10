import collections
import itertools
import math
import os
import pickle
import sqlite3
import datetime
from itertools import tee
from typing import Tuple, List, Iterator, Any, Counter, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from more_itertools import chunked
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report

import seaborn as sns
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.ndimage as ndi
import plotly.offline as offline
import plotly.graph_objs as go
import plotly
import mpl_toolkits.mplot3d.axes3d as p3
from imblearn.under_sampling import RandomUnderSampler
from math import sin, cos, acos, radians

# plotly.tools.set_credentials_file(username='<user name>', api_key='<api key>')


def connectDB():
    # CONNECT TO DB
    db_name      = "ais-research"
    db_name_path = "/media/ijs-aizu/ssd1/%s.db" % db_name
    connection   = sqlite3.connect(db_name_path)  # `connection` object is used as a database.
    print('Connected to DB.')

    return connection


# from: https://stackoverflow.com/questions/5878403/python-equivalent-to-rubys-each-cons
def pairwise(pairwise_list: Iterator[Any]) -> Iterator[Tuple[Any, Any]]: # [0, 1, 2, 3, 4, ...] -> [(0, 1), (1, 2), (2, 3), (3, 4), ...]
    a, b = tee(pairwise_list)  # [0, 1, 2, 3, 4, ...] -> a = [0, 1, 2, 3, 4, ...], b = [0, 1, 2, 3, 4, ...]
    _    = next(b, None)  # b = [1, 2, 3, 4, 5, ...]のように一個ずれる。_には0が入っている

    return zip(a, b)


def n_wise(n_wise_list: Iterator[Any], length: int = 2, overlap: int = 1) -> Iterator[Tuple[Any, ...]]:  # [0, 1, 2, 3, 4, ...] -> [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), ...]
    it = iter(n_wise_list)
    results = list(itertools.islice(it, length))
    while len(results) == length:
        yield tuple(results)
        results = results[length - overlap:]
        results.extend(itertools.islice(it, length - overlap))


def coodsToDegree(two_pair_coords: Tuple[List[Any], List[Any]]) -> Optional[float]:  # ([34.8, 127.6, timedelta(hoge)], [34.4, 128.0, timedelta(hoge)]) -> 314.0 or None
    coods1: List[Any] = two_pair_coords[0]
    coods2: List[Any] = two_pair_coords[1]
    delta_y: float    = coods2[1] - coods1[1]
    delta_x: float    = coods2[0] - coods1[0]
    delta_t           = coods2[2] - coods1[2]

    if delta_t > datetime.timedelta(days=1):
        return None

    if delta_x == 0 and delta_y > 0:
        return 270.0
    elif delta_x == 0 and delta_y < 0:
        return 90.0
    elif delta_x == 0 and delta_y == 0:
        return -1

    a = delta_y / delta_x
    degree = math.degrees(math.atan(a))
    if delta_x > 0 and delta_y > 0:
        pass
    elif delta_x < 0 and delta_y > 0:
        degree = 90 - degree
    elif delta_x < 0 and delta_y < 0:
        degree = 180 + degree
    elif delta_x > 0 and delta_y < 0:
        degree = 270 - degree

    # if degree < 0:
        # degree += 360.0

    # print(degree)
    return degree


def degreeToLevel(degree: float, n_of_degrees_level: int) -> int:
    degrees_per_level = 360 / n_of_degrees_level
    if degree == -1:  # n_of_degrees_levelが16のとき、0-15が方位のパラメータ、16が角度情報なし
        levels_in_one_year = n_of_degrees_level
    else:
        levels_in_one_year = int(degree / degrees_per_level)

    return levels_in_one_year


# from: https://qiita.com/s-wakaba/items/e12f2a575b6885579df7
def latlng_to_xyz(lat: float, lng: float) -> Tuple[float, float, float]:
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat * cos(rlng), coslat * sin(rlng), sin(rlat)


# from: https://qiita.com/s-wakaba/items/e12f2a575b6885579df7
def dist_on_sphere(pos0: Tuple[float, float], pos1: Tuple[float, float]) -> float:
    radius     = 6378.137  # Earth radius
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    sumension  = sum(x * y for x, y in zip(xyz0, xyz1))

    if abs(1 - sumension) < 0.0000001:
        sumension = 1.0
    elif abs(-1 - sumension) < 0.0000001:
        sumension = -1.0

    return acos(sumension) * radius


def get_velosities(times, latitudes: List[float], longitudes: List[float]) -> List[float]:
    pair_of_2points_list: List[Any] = list(n_wise(zip(times, zip(latitudes, longitudes)), length=2, overlap=1))
    distance_list = np.array([])

    for pair_of_2points in pair_of_2points_list:
        point_1       = pair_of_2points[0][1]
        point_2       = pair_of_2points[1][1]
        distance      = dist_on_sphere(point_1, point_2)
        distance_list = np.append(distance_list, distance)

    # print(list(distance_list))
    velosities = np.array([])

    for time_t1_t2, distance in zip(n_wise(times, length=2, overlap=1), distance_list):
        t1 = time_t1_t2[0]
        t2 = time_t1_t2[1]
        delta_t = t2 - t1

        if delta_t.total_seconds() < 0:
            print('Incorrect time series:', delta_t.total_seconds())
            exit()

        delta_t_hours = delta_t.total_seconds() / 60 / 60
        if delta_t_hours == 0:
            if distance == 0:
                velosity = 0
            else:
                velosity = -1
        else:
            velosity = distance / delta_t_hours

        velosities = np.append(velosities, velosity)

    return velosities


def get_degrees_collection_counter(pairwised_coords: List[Any]) -> List[int]:  # [([31.7, 125.2], [32.0, 125.4]), ...] -> (17, 17)などのペアの出現回数リスト[14, 15, 5, 5, 0, 0, ...]
    degrees_in_one_year: Iterator[float] = filter(lambda x: x is not None, map(lambda x: coodsToDegree(x), pairwised_coords))
    n_pair: int                 = 2
    n_of_degrees_level: int     = 16  # 例: 16なら 0-15の16段階の角度情報と角度情報なし16の計17レベル
    n_dim_labels: Iterator[int] = map(lambda x: degreeToLevel(x, n_of_degrees_level), degrees_in_one_year)  # [16, 16, 12, 16,...]のように1隻の一年分のパラメータ化された値の配列
    pairwised_n_dim_labels      = list(n_wise(n_dim_labels, length=n_pair, overlap=n_pair - 1))  # [(16, 16), (16, 12), (12, 16),...]のように前から2つずつタプル化。これがhashのkeyになる
    collections_counter_hash: Counter[Tuple[Any, ...]] = collections.Counter(pairwised_n_dim_labels)  # {(16, 16): 893, (16, 12): 63, (12, 16): 60,...}のように出現数カウント　collections.Counter(['a', 'a', 'b']) -> Counter({'a': 2, 'b': 1})
    seq                         = range(0, n_of_degrees_level + 1)  # [0, 1, 2,..., 16]
    degrees_collections_counter: List[int] = []

    for pair in itertools.product(seq, repeat=n_pair):  # このforループで{(16, 16): 893, (16, 15): 8, (16, 14): 25,...} のように全部パターンの出現数のkeyをhashに追加した
        collections_counter_hash[pair] = collections_counter_hash.get(pair, 0)  # 例えばkey(10, 3)を探し、存在しなかったら{(10, 3): 0}を追加する

    for value in itertools.product(seq, repeat=n_pair):  # {(0, 0): 12, (0, 1): 4, (0, 2): 6,...} -> [12, 4, 6,...]のようにvalueのみ取り出し、リストに入れる
        degrees_collections_counter.append(collections_counter_hash.get(value, 0))
    # degrees_collections_counter = list(collections_counter_hash.values())  # 上2行はこれ1行で書き直せるけど、多い順ソート？されてしまうので後半0,0,0,...ばかりになってしまう。テスト分割時に情報0になってしまって精度下がる気がする

    return degrees_collections_counter


def get_v(x: float) -> int:
    v = int(x)

    # if -1 < v < 25:
    #     v = 1
    # elif v <= 50:
    #     v = 2
    # elif v > 50:
    #     v = 3
    # else:
    #     v = -1
    if v > 30:
        v = 31

    return v


def get_velosities_collection_counter(velosities: List[float]) -> List[int]:
    n_dim = 33  # -1, 0~30, 31
    hash: Dict[int, int] = {}
    velosities_int = map(lambda x: get_v(noneToNum(x)), velosities)
    hash = collections.Counter(velosities_int)
    velosities_collection_counter = []

    for i in range(-1, n_dim - 1):
        hash[i] = hash.get(i, 0)  # 例えばkey=3を探し、存在しなかったら{3: 1}を追加する

    for value in range(-1, n_dim - 1):
        velosities_collection_counter.append(hash.get(value, 0))
    return velosities_collection_counter


def noneToNum(x: Any) -> int:
    if isinstance(x, float):
        return int(x)
    else:
        return -1


def get_cogs_collection_counter2(congs: List[Any]) -> List[int]:
    #degrees_in_one_year: Iterator[float] = filter(lambda x: x is not None, map(lambda x: coodsToDegree(x), congs))
    n_pair: int                 = 2
    n_of_degrees_level: int     = 16  # 例: 16なら 0-15の16段階の角度情報と角度情報なし16の計17レベル
    #n_dim_labels: Iterator[int] = map(lambda x: degreeToLevel(x, n_of_degrees_level), degrees_in_one_year)  # [16, 16, 12, 16,...]のように1隻の一年分のパラメータ化された値の配列
    n_dim_labels: Iterator[int] = get_cogs_collection_counter(congs)
    pairwised_n_dim_labels      = list(n_wise(n_dim_labels, length=n_pair, overlap=n_pair - 1))  # [(16, 16), (16, 12), (12, 16),...]のように前から2つずつタプル化。これがhashのkeyになる
    collections_counter_hash: Counter[Tuple[Any, ...]] = collections.Counter(pairwised_n_dim_labels)  # {(16, 16): 893, (16, 12): 63, (12, 16): 60,...}のように出現数カウント　collections.Counter(['a', 'a', 'b']) -> Counter({'a': 2, 'b': 1})
    seq                         = range(0, n_of_degrees_level + 1)  # [0, 1, 2,..., 16]
    degrees_collections_counter: List[int] = []

    for pair in itertools.product(seq, repeat=n_pair):  # このforループで{(16, 16): 893, (16, 15): 8, (16, 14): 25,...} のように全部パターンの出現数のkeyをhashに追加した
        collections_counter_hash[pair] = collections_counter_hash.get(pair, 0)  # 例えばkey(10, 3)を探し、存在しなかったら{(10, 3): 0}を追加する

    for value in itertools.product(seq, repeat=n_pair):  # {(0, 0): 12, (0, 1): 4, (0, 2): 6,...} -> [12, 4, 6,...]のようにvalueのみ取り出し、リストに入れる
        degrees_collections_counter.append(collections_counter_hash.get(value, 0))
    # degrees_collections_counter = list(collections_counter_hash.values())  # 上2行はこれ1行で書き直せるけど、多い順ソート？されてしまうので後半0,0,0,...ばかりになってしまう。テスト分割時に情報0になってしまって精度下がる気がする

    return degrees_collections_counter

def get_cogs_collection_counter(cogs: List[float]) -> List[int]:
    n_dim = 17  # -1, 0~15
    hash: Dict[int, int] = {}
    cogs_int = map(lambda x: degreeToLevel(noneToNum(x), 16), cogs)
    hash = collections.Counter(cogs_int)
    cogs_collection_counter = []

    for i in range(-1, n_dim - 1):
        hash[i] = hash.get(i, 0)  # 例えばkey=3を探し、存在しなかったら{3: 1}を追加する

    for value in range(-1, n_dim - 1):
        cogs_collection_counter.append(hash.get(value, 0))

    return cogs_collection_counter


def get_rots_collection_counter(rots: List[float]) -> List[float]:
    hash: Dict[float, int] = {}
    rots_filtered = filter(lambda x: x != 'None', filter(lambda x: (x != '') and not None, rots))
    hash = collections.Counter(rots_filtered)
    common_n = 5
    if len(hash) < common_n:
        return list(np.ones((common_n * 2, 1), dtype=float) * 100)
    rots_collection_counter = list(hash.keys())[:common_n] + list(map(lambda x: float(x), list(hash.values())[:common_n]))

    return rots_collection_counter


def createFeatureVector(times: List[Any], latitudes: List[Any], longitudes: List[Any], rots: List[Any], sogs: List[Any], cogs: List[Any], headings: List[Any]) -> List[Any]:
    n_pair: int                      = 2
    ship_t_lat_lon: Iterator[Any]    = map(lambda x: list(x), zip(times, latitudes, longitudes))  # [[datetime.datetime(2017, 12, 16, 1, 58, 47), 34.9366666667, 129.065], ...]
    sorted_ship_t_lat_lon: List[Any] = sorted(ship_t_lat_lon, key=lambda x: x[0])
    pairwised_coords: List[Any]      = list(n_wise(map(lambda x: [x[1], x[2], x[0]], sorted_ship_t_lat_lon), length=n_pair, overlap=n_pair - 1))  # 渡したリストの要素を1個ずつ取り出す。 x[0] -> datetime.datetime(2017, 12, 16, 1, 58, 47), x[1] -> 34.9366666667, x[2] -> 129.065]
    # 2点間の角度: degree
    degrees_collections_counter: List[int]          = get_degrees_collection_counter(pairwised_coords)
    # print(degrees_collections_counter)
    # print(len(degrees_collections_counter))
    degrees_collections_counter_log: List[float] = list(np.log(np.array(degrees_collections_counter) + 1))  # ヒストグラム(出現回数)のlogを取る時、log(0)は-infになってしまうので+1している
   
    # 2点間の速度: velosities
    velosities: List[float]                         = get_velosities(times, latitudes, longitudes)
    velosities_collections_counter: List[int]       = get_velosities_collection_counter(velosities)
    velosities_collections_counter_log: List[float] = list(np.log(np.array(velosities_collections_counter) + 1))  # ヒストグラム(出現回数)のlogを取る時、log(0)は-infになってしまうので+1している

    # ROT
    # rots_collections_counter: List[float] = get_rots_collection_counter(rots)

    # 対地速度: Speed Over Ground
    sogs_collections_counter: List[int] = get_velosities_collection_counter(sogs)
    sogs_collections_counter_log: List[float] = list(np.log(np.array(sogs_collections_counter) + 1))  # ヒストグラム(出現回数)のlogを取る時、log(0)は-infになってしまうので+1している

    # 対地方位角度: Course Over Ground
    cogs_collections_counter: List[int] = get_cogs_collection_counter2(cogs)
    cogs_collections_counter_log: List[float] = list(np.log(np.array(cogs_collections_counter) + 1))  # ヒストグラム(出現回数)のlogを取る時、log(0)は-infになってしまうので+1している

    # 進行方位角度: Heading
    headings_collections_counter: List[int] = get_cogs_collection_counter2(headings)
    headings_collections_counter_log: List[float] = list(np.log(np.array(headings_collections_counter) + 1))  # ヒストグラム(出現回数)のlogを取る時、log(0)は-infになってしまうので+1している

    # feature_vector = np.array(degrees_collections_counter_log + velosities_collections_counter_log + sogs_collections_counter_log + cogs_collections_counter_log + headings_collections_counter_log)  # なんで速度だけlog取ってるのか忘れた...
    feature_vector = np.array(degrees_collections_counter_log + velosities_collections_counter_log)
    # feature_vector = np.array(degrees_collections_counter_log + velosities_collections_counter_log + rots_collections_counter + sogs_collections_counter_log + cogs_collections_counter_log + headings_collections_counter_log)
    # print(len(feature_vector))
    # exit()
    return feature_vector


def setShipTypeLabel(mmsi: int):
    ship_type_df = pd.read_sql_query('SELECT Ship_Type from csv_remove_n_and_distinct_mmsi_and_having_lt2 WHERE MMSI=:mmsi', con=connection, params={'mmsi': str(mmsi)})
    label        = ship_type_df['Ship_Type'][0]
    # label        = ship_type_df['ship_type_manually_added'][0]

    return label


def plot_velosities_hist(ships_velosities: List[float]):
    plt.hist(ships_velosities, range=(1, 50), bins=49)
    plt.show()
    plt.close()


def generateFeatureVector(connection, feature_vec_description: str) -> Tuple[List[List[int]], List[str]]:  # ([[0, 3, 16, ...], [0, 0, 9, ...], ...], ['Cargo', 'Container Ship', ...])
    print('feature_vec_description:', feature_vec_description)
    feature_vecs: List[Any]            = []  # [[0, 3, 16, ...], [0, 0, 9, ...], ...]
    labels: List[str]                  = []  # ['Cargo', 'Container Ship', ...]
    pickle_file: str                   = 'df_list_191204_3class_limit_50_ships_with_velosities.pickle'

    all_ships_1year_df_list: List[Any] = []  # [df, df,...]

    # 使用するDBファイルのpickleを作成
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            all_ships_1year_df_list = pickle.load(f)
        print('Pickle Loaded.')

    else:
        # 学習に使う船のMMSI一覧を取得する
        # 例: 船種がcargo_ships, tankships, fishing_boatsのMMSIのリスト
        # 識別する船種の数を増やしたい時はこことclassifierの`np.zeros()`を変更する
        mmsi_df_1 = pd.read_sql_query("SELECT a.MMSI MMSI, count(*) cnt FROM csv_remove_n_and_distinct_mmsi_and_having_lt2 a, for_research b WHERE a.MMSI = b.MMSI and a.Ship_Type='cargo_ships' GROUP BY a.MMSI ORDER BY cnt DESC LIMIT 50;", con=connection)
        mmsi_df_2 = pd.read_sql_query("SELECT a.MMSI MMSI, count(*) cnt FROM csv_remove_n_and_distinct_mmsi_and_having_lt2 a, for_research b WHERE a.MMSI = b.MMSI and a.Ship_Type='fishing_boats' GROUP BY a.MMSI ORDER BY cnt DESC LIMIT 50;", con=connection)
        mmsi_df_3 = pd.read_sql_query("SELECT a.MMSI MMSI, count(*) cnt FROM csv_remove_n_and_distinct_mmsi_and_having_lt2 a, for_research b WHERE a.MMSI = b.MMSI and a.Ship_Type='passenger_ships' GROUP BY a.MMSI ORDER BY cnt DESC LIMIT 50;", con=connection)
        # mmsi_df_4 = pd.read_sql_query("SELECT a.MMSI MMSI, count(*) cnt FROM csv_remove_n_and_distinct_mmsi_and_having_lt2 a, for_research b WHERE a.MMSI = b.MMSI and a.Ship_Type='tankships' GROUP BY a.MMSI ORDER BY cnt DESC LIMIT 50;", con=connection)
        # mmsi_df_5 = pd.read_sql_query("SELECT a.MMSI MMSI, count(*) cnt FROM csv_remove_n_and_distinct_mmsi_and_having_lt2 a, for_research b WHERE a.MMSI = b.MMSI and a.Ship_Type='tugboats' GROUP BY a.MMSI ORDER BY cnt DESC LIMIT 100;", con=connection)

        # mmsi_df_1 = pd.read_sql_query("SELECT MMSI FROM ship_type WHERE Ship_Type='cargo_ships' LIMIT 200;", con=connection)
        # mmsi_df_2 = pd.read_sql_query("SELECT MMSI FROM ship_type WHERE Ship_Type='tankships' LIMIT 200;", con=connection)
        # mmsi_df_3 = pd.read_sql_query("SELECT MMSI FROM ship_type WHERE Ship_Type='fishing_boats' LIMIT 200;", con=connection)
        mmsi_df_usinglist   = [mmsi_df_1, mmsi_df_2, mmsi_df_3]
        n_of_shiptypes: int = len(mmsi_df_usinglist)  # FIXME: 配列の要素の数ではなく、dfの中のship_typeの種類で数をカウントするべき
        if n_of_shiptypes > 1:
            mmsi_df           = pd.concat(mmsi_df_usinglist)
        else:
            print('Please set more than 1 ship type:', n_of_shiptypes)
            exit()

        # 船種がFishing, Cargo, ShipのMMSIのリスト
        # mmsi_df = pd.read_sql_query("SELECT MMSI FROM mmsi WHERE ship_class='Fishing' OR ship_class='Cargo' OR ship_class='Container Ship' LIMIT :limit;", con=connection, params={'limit': 1000})
        # データが多い船種上位n件のMMSIのリスト ['Bulk Carrier' 'Cargo' 'Container Ship' 'Fishing' 'Oil/Chemical Tanker']
        # mmsi_df = pd.read_sql_query("SELECT mmsi FROM mmsi WHERE ship_class IN (SELECT ship_class FROM mmsi WHERE (ship_class IS NOT NULL AND ship_class != 'Other' AND ship_class != 'No Records Found') GROUP BY ship_class HAVING count(*) > 10);", con=connection)

        for df in mmsi_df.iterrows():  # [df, df,...] を作る。一つのdfにはは一隻の一年間の[mmsi, Time, Latitude, Longitude]が格納されている
            one_ship_mmsi: int    = df[1]['MMSI']  # 23481047みたいなMMSI一隻分
            mmsi_time_df          = pd.read_sql_query("SELECT mmsi, Time, Latitude, Longitude, ROT, SOG, COG, Heading FROM for_research WHERE mmsi=:mmsi ORDER BY Time ASC;", con=connection, params={'mmsi': str(one_ship_mmsi)})  # 一隻の一年分の[mmsi, Time, Latitude, Longitude]を取得 # TODO: ここの構造を変えるときはピクルを作り直すこと
            all_ships_1year_df_list.append(mmsi_time_df)  # 一隻の一年分のデータのdataframeをlistに加えていく

    # velosities_master = np.array([])
    # 一隻の一年間分のデータを取り出して加工し、feature_vecsに追加していく
    temp = True
    for one_ship in all_ships_1year_df_list:
        times: List[Any]      = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), list(one_ship['Time'])))  # [datetime.datetime(2017, 1, 1, 13, 58, 47), ...]
        latitudes: List[Any]  = list(one_ship['Latitude'])   # [33.0433333333, 34.9366666667, ...]
        longitudes: List[Any] = list(one_ship['Longitude'])  # [127.011666667, 126.643333333, ...]
        rots: List[Any]       = list(one_ship['ROT'])  # [1.11600720834, 0, 0, ...]
        sogs: List[Any]       = list(one_ship['SOG'])  # [13, 12, 13, 0, ...]
        cogs: List[Any]       = list(one_ship['COG'])  # [49, 51, 51, 51, ...]
        headings: List[Any]   = list(one_ship['Heading'])  # [235, 130, 272, ...]
        mmsi: int             = one_ship['MMSI'][0]  # 538004893

        feature_vec = createFeatureVector(times, latitudes, longitudes, rots, sogs, cogs, headings)
        label       = setShipTypeLabel(mmsi)
        feature_vecs.append(feature_vec)
        labels.append(label)
        # surface_plot(mmsi, times, latitudes, longitudes)

        # all_ships_velosities = np.append(velosities_master, velosities)
    # plot_velosities_hist(all_ships_velosities)
        if temp:
            print(len(feature_vec))
            temp = False

    if not os.path.exists(pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_ships_1year_df_list, f)
            print('Pickle Dumped.')

    return (feature_vecs, labels)


def surface_plot(mmsi, times, latitudes, longitudes):
    ship_t_lat_lon: Iterator[Any]    = map(lambda x: list(x), zip(times, latitudes, longitudes))  # [[datetime.datetime(2017, 12, 16, 1, 58, 47), 34.9366666667, 129.065], ...]
    sorted_ship_t_lat_lon: List[Any] = sorted(ship_t_lat_lon, key=lambda x: x[0])
    pairwised_coords: List[Any]      = list(n_wise(map(lambda x: [x[1], x[2]], sorted_ship_t_lat_lon), length=2, overlap=1))  # 渡したリストの要素を1個ずつ取り出す。 x[0] -> datetime.datetime(2017, 12, 16, 1, 58, 47), x[1] -> 34.9366666667, x[2] -> 129.065]

    # degrees_collections_counter: List[int] = get_degrees_collection_counter(pairwised_coords)
    degrees_in_one_year: Iterator[float] = map(lambda x: coodsToDegree(x), pairwised_coords)

    n_of_degrees_level: int     = 16  # 例: 16なら 0-15の16段階の角度情報と角度情報なし16の計17レベル
    n_dim_labels: Iterator[int] = map(lambda x: degreeToLevel(x, n_of_degrees_level), degrees_in_one_year)  # [16, 16, 12, 16,...]のように1隻の一年分のパラメータ化された値の配列
    pairwised_n_dim_labels      = list(n_wise(n_dim_labels))  # [(16, 16), (16, 12), (12, 16),...]のように前から2つずつタプル化。これがhashのkeyになる
    collections_counter_hash: Counter[Tuple[Any, ...]] = collections.Counter(pairwised_n_dim_labels)  # {(16, 16): 893, (16, 12): 63, (12, 16): 60,...}のように出現数カウント collections.Counter(['a', 'a', 'b']) -> Counter({'a': 2, 'b': 1})
    seq                         = range(0, n_of_degrees_level + 1)  # [0, 1, 2,..., 16]

    for pair in itertools.product(seq, repeat=2):  # このforループで{(16, 16): 893, (16, 15): 8, (16, 14): 25,...} のように全部パターンの出現数のkeyをhashに追加した
        collections_counter_hash[pair] = collections_counter_hash.get(pair, 0)  # 例えばkey(10, 3)を探し、存在しなかったら{(10, 3): 0}を追加する

    X = []
    Y = []
    Z = []
    for xy, z in collections_counter_hash.items():
        X.append(xy[0])
        Y.append(xy[1])
        Z.append(z)
    # X = np.array(X).reshape(17, 17)
    # Y = np.array(Y).reshape(17, 17)
    # Z = np.array(Z).reshape(17, 17)

    fig    = plt.figure(figsize=(8, 6))
    ax1    = fig.add_subplot(111, projection='3d')
    top    = np.array(Z)
    bottom = np.zeros_like(Z)
    width  = depth = 1

    ax1.bar3d(X, Y, bottom, width, depth, top, shade=True, alpha=0.6, color='#1dF144')
    ax1.set_xlabel('t')
    ax1.set_ylabel('t-1')
    ax1.set_zlabel('# of records')
    ax1.set_xlim([0, 16])
    ax1.set_ylim([0, 16])
    ax1.set_zlim([0, 64])
    ship_type_df = pd.read_sql_query('SELECT ship_class from mmsi WHERE MMSI=:mmsi', con=connection, params={'mmsi': str(mmsi)})
    ship_type    = ship_type_df['ship_class'][0]
    ax1.set_title('MMSI: %s\nShip_Class: %s' % (mmsi, ship_type))
    sns.set()
    today = str(datetime.today())[:-16]
    bar3d_filename = '%s-%s-%s' % (today, ship_type, mmsi)
    plt.savefig('images/_%s.png' % bar3d_filename)
    plt.close()
    # plt.show()

    # まあまあ
    # fig = plt.figure()
    # ax = p3.Axes3D(fig)
    # ax.contourf3D(X, Y, Z)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # fig.add_axes(ax)
    # plt.show()

    # びみょう
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(np.ravel(X), np.ravel(Y), Z)
    # ax.set_title("Scatter Plot")
    # plt.show()

    # ぐちゃった
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='plasma', rstride=3, cstride=3)
    # plt.show()

    return

# def classifier(feature_vecs: list, labels: list):
#     ## classify 3-fold cross validation 5 times ##
#     print(feature_vecs, labels)


def classifier(feature_vecs: List[List[int]], labels: List[str]):
    models        = ['MLP', 'SVM']
    # models['MLP'] = MLPClassifier(solver='adam', random_state=0, max_iter=10000)
    # models['MLP'] = MLPClassifier()
    # models['SVM'] = SVC()
    # models['KNC'] = KNeighborsClassifier()
    # models['DTC'] = DecisionTreeClassifier()
    print("==================")
    labels = list(map(lambda x: 'None' if x is None else x, labels))  # 前処理: データベースのラベルが`None`だと落ちるので文字列の'None'に変換する

    X  = np.array(feature_vecs)
    le = LabelEncoder()  # scikit-learnはラベルが文字列だと落ちるので数字に変換する
    y  = np.array(le.fit_transform(labels))

    # ref: http://pynote.hatenablog.com/entry/sklearn-feature-scaling
    transformers = {}
    transformers['std']        = StandardScaler()
    # transformers['min-max']    = MinMaxScaler()
    # transformers['max-abs']    = MaxAbsScaler()
    # transformers['robust']     = RobustScaler()
    transformers['normalizer'] = Normalizer()

    for transformer_name, transformer in transformers.items():
        print("【Transformer】: %s" % transformer_name)
        print("==================")

        if False:
            if transformer_name == 'std':
                plt.figure(figsize=(5, 6))
                plt.subplot(2, 1, 1)
                plt.title('StandardScaler')
                plt.xlim([-4, 10])
                plt.ylim([-4, 10])
                plt.scatter(X_orig[:, 0], X_orig[:, 1], c='red', marker='x', s=30, label='origin')
                plt.scatter(X[:, 0], X[:, 1], c='blue', marker='x', s=30, label='standard ')
                plt.legend(loc='upper left')
                plt.hlines(0, xmin=-4, xmax=10, colors='#888888', linestyles='dotted')
                plt.vlines(0, ymin=-4, ymax=10, colors='#888888', linestyles='dotted')
                # plt .show()

            elif transformer_name == 'min-max':
                plt.subplot(2, 1, 2)
                plt.title('MinMaxScaler')
                plt.xlim([-4, 10])
                plt.ylim([-4, 10])
                plt.scatter(X_orig[:, 0], X_orig[:, 1], c='red', marker='x', s=30, label='origin')
                plt.scatter(X[:, 0], X[:, 1], c='green', marker='x', s=30, label='normalize')
                plt.legend(loc='upper left')
                plt.hlines(0, xmin=-4, xmax=10, colors='#888888', linestyles='dotted')
                plt.vlines(0, ymin=-4, ymax=10, colors='#888888', linestyles='dotted')
                # plt.show()

            else:
                plt.figure(figsize=(5, 6))
                plt.subplot(2, 1, 1)
                plt.title(transformer_name)
                plt.xlim([-4, 10])
                plt.ylim([-4, 10])
                plt.scatter(X_orig[:, 0], X_orig[:, 1], c='red', marker='x', s=30, label='origin')
                plt.scatter(X[:, 0], X[:, 1], c='blue', marker='x', s=30, label='standard ')
                plt.legend(loc='upper left')
                plt.hlines(0, xmin=-4, xmax=10, colors='#888888', linestyles='dotted')
                plt.vlines(0, ymin=-4, ymax=10, colors='#888888', linestyles='dotted')
                # plt.show()

        k_fold = 5
        for model_name in models:
            print("Model:   %s" % model_name)

            # sum_best_cmx = np.zeros((3, 3))
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=821)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                transformer.fit(X_train)
                X_train_scaled  = transformer.transform(X_train)
                X_test_scaled   = transformer.transform(X_test)
                X_train, X_test = X_train_scaled, X_test_scaled

                if model_name == 'SVM':
                    tuned_parameters = [
                        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
                        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
                        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
                    ]
                    model = GridSearchCV(
                        SVC(random_state=821),
                        tuned_parameters,
                        cv=5,
                        iid=False,
                        scoring='accuracy'
                    )

                elif model_name == 'MLP':
                    tuned_parameters = [
                        {"hidden_layer_sizes": [(100,), (200,), (600,)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], },
                        {"hidden_layer_sizes": [(100, 100), (200, 100)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], }
                        # {"hidden_layer_sizes": [(1000,), (2000,), (6000,)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], },
                        # {"hidden_layer_sizes": [(1000, 1000), (2000, 1000)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], }
                    ]
                    model = GridSearchCV(
                        MLPClassifier(),
                        tuned_parameters,
                        cv=5,
                        iid=False,
                        scoring='accuracy'
                    )

                model.fit(X_train, y_train)
                best_y_pred = model.predict(X_test)  # Predict on the estimator with the best found parameters.
                best_cmx         = confusion_matrix(y_test, best_y_pred)
                # sum_best_cmx    += best_cmx

                # print('Classification Report : {}'.format(classification_report(y_test, best_y_pred)))
                # print('Model parameters      : {}'.format(model.get_params))
                print('Best parameters       : {}'.format(model.best_params_))
                print('Best cross-validation : {}'.format(model.best_score_))
                print('Mean Best cv          : {}'.format(model.best_score_))
                # print('Best Estimeter        : {}'.format(model.best_estimator_))  # best_params_を含む
                print('Test set score        : {}'.format(model.score(X_test, y_test)))
                print('Test confusion matrix :\n', best_cmx)
            # print('5 times sum of best confusion matrix :\n', sum_best_cmx)  # いろんなモデルが混ざってしまっているので不適切


def best_param_classifier(feature_vecs: List[List[int]], labels: List[str]):
    models        = {}
    # models['MLP'] = MLPClassifier(solver='adam', random_state=0, max_iter=10000)
    models['MLP'] = MLPClassifier(
        activation='relu',
        early_stopping=True,
        hidden_layer_sizes=(200, 100),
        max_iter=1000,
        random_state=12345,
        solver='lbfgs'
    )
    models['SVM'] = SVC(
        C=1000,
        gamma=0.001,
        kernel='rbf'
    )
    # models['KNC'] = KNeighborsClassifier()
    # models['DTC'] = DecisionTreeClassifier()
    labels = list(map(lambda x: 'None' if x is None else x, labels))  # 前処理: データベースのラベルが`None`だと落ちるので文字列の'None'に変換する

    X  = np.array(feature_vecs)
    le = LabelEncoder()  # scikit-learnはラベルが文字列だと落ちるので数字に変換する
    y  = np.array(le.fit_transform(labels))
    print(le.inverse_transform([0, 1, 2]))

    # ref: http://pynote.hatenablog.com/entry/sklearn-feature-scaling
    transformers = {}
    transformers['std']        = StandardScaler()
    transformers['min-max']    = MinMaxScaler()
    transformers['max-abs']    = MaxAbsScaler()
    # transformers['robust']     = RobustScaler()
    # transformers['normalizer'] = Normalizer()

    for transformer_name, transformer in transformers.items():
        k_fold  = 5
        n_times = 5
        for model_name, model in models.items():
            print("========================")
            print("【Transformer】: %s" % transformer_name)
            print("【   Model   】: %s" % model_name)
            print("========================")

            sum_cmx  = np.zeros((3, 3), dtype=int)
            ave_acc = 0.0
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=821)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                transformer.fit(X_train)
                X_train_scaled  = transformer.transform(X_train)
                X_test_scaled   = transformer.transform(X_test)
                X_train, X_test = X_train_scaled, X_test_scaled

                model.fit(X_train, y_train)
                best_y_pred = model.predict(X_test)
                cmx         = confusion_matrix(y_test, best_y_pred)
                sum_cmx    += cmx
                acc         = model.score(X_test, y_test)
                ave_acc    += acc

                # print('Classification Report : {}'.format(classification_report(y_test, best_y_pred)))
                # print('Model parameters      : {}'.format(model.params_))
                print('Test set score : {}'.format(acc))
                # print('Test confusion matrix :\n', cmx)
            ave_acc /= n_times
            print("------------------------")
            print('5 times sum of confusion matrix :\n', sum_cmx)
            print('Ave-Acc:', ave_acc)
            print()


# from: Pythonで始める機械学習
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    # for each split of the data in the outer cross-validation
    # (split method returns indices)
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter using inner cross-validation
        best_params = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # accumulate score over inner splits
            cv_scores = []
            # iterate over inner cross-validation
            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
                # build classifier given parameters and training data
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test set
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            # compute mean score over inner folds
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score
                best_params = parameters
        # build classifier on best parameters using outer training set
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)


def _classifier(feature_vecs: List[List[int]], labels: List[str]):
    models        = ['MLP']#, 'SVM']

    print("==================")
    labels = list(map(lambda x: 'None' if x is None else x, labels))  # 前処理: データベースのラベルが`None`だと落ちるので文字列の'None'に変換する

    X  = np.array(feature_vecs)
    le = LabelEncoder()  # scikit-learnはラベルが文字列だと落ちるので数字に変換する
    y  = np.array(le.fit_transform(labels))

    # ref: http://pynote.hatenablog.com/entry/sklearn-feature-scaling
    transformers = {}
    transformers['std']        = StandardScaler()
    transformers['normalizer'] = Normalizer()

    for transformer_name, transformer in transformers.items():
        print("【Transformer】: %s" % transformer_name)
        X_orig = X
        X      = transformer.fit_transform(X_orig)
        print("==================")

        for model_name in models:
            print("Model:   %s" % model_name)

            if model_name == 'SVM':
                tuned_parameters = [
                    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
                    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
                    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
                ]
                model = GridSearchCV(
                    SVC(random_state=821),
                    tuned_parameters,
                    cv=5,
                    iid=False,
                    scoring='accuracy',
                )
                scores = cross_val_score(model, X, y, cv=5)
                print('cross-validation scores:', scores)
                print('model.best_params_     :', model.best_params_)
                print('model.best_score_      :', model.best_score_)

                # y_pred        = model.predict(X_test)
                # test_accuracy = model.score(X_test, y_test)
                # cmx           = confusion_matrix(y_test, y_pred)

            elif model_name == 'MLP':
                tuned_parameters = [
                    {"hidden_layer_sizes": [(100,), (200,), (600,)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], },
                    {"hidden_layer_sizes": [(100, 100), (200, 100)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], }
                    # {"hidden_layer_sizes": [(1000,), (2000,), (6000,)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], },
                    # {"hidden_layer_sizes": [(1000, 1000), (2000, 1000)], "activation": ['relu'], "solver": ["lbfgs", "adam"], "max_iter": [1000], "early_stopping":[True], "random_state":[12345], }
                ]
                model = GridSearchCV(
                    MLPClassifier(random_state=12345),
                    tuned_parameters,
                    cv=5,
                    iid=False,
                    scoring='accuracy'
                )
                scores = cross_val_score(model, X, y, cv=5)

                # y_pred        = model.predict(X_test)
                # test_accuracy = model.score(X_test, y_test)
                # print('model.predict(X_test)      : ', y_pred)
                # print('model.score(X_test, y_test):', test_accuracy)
                # cmx           = confusion_matrix(y_test, y_pred)

                # print('model.best_params_      :', model.best_params_)
                print('cross-validation scores :', scores)
                # print('best_estimator_         :', model.model_best_estimator)
                # print('model.best_score_       :', model.best_score_)

                # テストデータセットでの分類精度を表示
                # print("The scores are computed on the full evaluation set.")
                # print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    connection              = connectDB()
    feature_vec_description = '17-dim degree param in one year'
    feature_vecs, labels    = generateFeatureVector(connection, feature_vec_description)
    # classifier(feature_vecs, labels)
    best_param_classifier(feature_vecs, labels)