import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import random


def load_data(filepath):
    pokemons = []
    with open(filepath, newline='', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            number = int(row['#'])
            name = row['Name']
            type1 = row['Type 1']
            type2 = row['Type 2']
            total = int(row['Total'])
            hp = int(row['HP'])
            attack = int(row['Attack'])
            defense = int(row['Defense'])
            sp_atk = int(row['Sp. Atk'])
            sp_def = int(row['Sp. Def'])
            speed = int(row['Speed'])
            pokemon = {
                '#': number,
                'Name': name,
                'Type 1': type1,
                'Type 2': type2,
                'Total': total,
                'HP': hp,
                'Attack': attack,
                'Defense': defense,
                'Sp. Atk': sp_atk,
                'Sp. Def': sp_def,
                'Speed': speed,
            }
            pokemons.append(pokemon)
    first_20_pokemons = []
    for i in range(20):
        first_20_pokemons.append(pokemons[i])
    return first_20_pokemons


def calculate_x_y(stats):
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (x, y)


def hac(dataset):
    # pop out invalid data points, i.e. NaN or inf
    filter_dataset = []
    for i in dataset:
        valid = True
        for j in i:
            if math.isinf(j):    # filter inf
                valid = False
            elif math.isnan(j):  # filter NaN
                valid = False
        if valid:
            filter_dataset.append(i)

    m = len(filter_dataset)
    points = [i for i in range(m)]
    clusters = []
    z = []
    for i in range(m - 1):
        xi, xj, dist = closest_dist(filter_dataset, clusters, points)
        if xi <= (m - 1) and xj <= (m - 1):
            points.remove(xi)
            points.remove(xj)
            new_cluster = {'id': m + i, 'elements': [xi, xj], 'recently_created': True}
            clusters.append(new_cluster)
            z.append([xi, xj, dist, 2])
        elif xi <= (m - 1) and xj > (m - 1):
            points.remove(xi)
            index = get_index(clusters, xj)
            clusters[index]['recently_created'] = False
            new_cluster, count = combine_cluster_and_point(m + i, clusters[index], xi)
            clusters.append(new_cluster)
            z.append([xi, xj, dist, count])
        elif xi > (m - 1) and xj > (m - 1):
            index_i = get_index(clusters, xi)
            index_j = get_index(clusters, xj)
            clusters[index_i]['recently_created'] = False
            clusters[index_j]['recently_created'] = False
            new_cluster, count = combine_clusters(m + i, clusters[index_i], clusters[index_j])
            clusters.append(new_cluster)
            z.append([xi, xj, dist, count])

    return np.array(z)


# helper function: get closest pair among points
def closest_dist_between_points(dataset, indexes):
    min_distance = float('inf')
    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            distance = math.dist(dataset[indexes[i]], dataset[indexes[j]])
            if distance < min_distance:
                min_distance = distance
    min_pairs = []
    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            distance = math.dist(dataset[indexes[i]], dataset[indexes[j]])
            if distance == min_distance:
                if indexes[i] < indexes[j]:
                    min_pairs.append((indexes[i], indexes[j]))
                else:
                    min_pairs.append((indexes[j], indexes[i]))

    min_xi = float('inf')
    min_xj = float('inf')
    for pair in min_pairs:
        xi, xj = pair
        if xi < min_xi:
            min_xi = xi

    for pair in min_pairs:
        xi, xj = pair
        if xj < min_xj and xi == min_xi:
            min_xj = xj

    return min_xi, min_xj, min_distance


# helper function: get closest pair among clusters
def closest_dist_between_clusters(dataset, cluster_1, cluster_2):
    min_distance = float('inf')
    for i in cluster_1:
        for j in cluster_2:
            distance = math.dist(dataset[i], dataset[j])
            if distance < min_distance:
                min_distance = distance
    min_pairs = []
    for i in cluster_1:
        for j in cluster_2:
            distance = math.dist(dataset[i], dataset[j])
            if distance == min_distance:
                if i < j:
                    min_pairs.append((i, j))
                else:
                    min_pairs.append((j, i))

    min_xi = float('inf')
    min_xj = float('inf')
    for pair in min_pairs:
        xi, xj = pair
        if xi < min_xi:
            min_xi = xi
    for pair in min_pairs:
        xi, xj = pair
        if xj < min_xj and xi == min_xi:
            min_xj = xj
    return min_xi, min_xj, min_distance


# helper function: get closest pair
def closest_dist(dataset, clusters, points):
    recently_created_clusters = []
    for cluster in clusters:
        if cluster['recently_created']:
            recently_created_clusters.append(cluster)

    pairs = []
    # get closest pair among points
    a_i, a_j, a_dist = closest_dist_between_points(dataset, points)
    pairs.append((a_i, a_j, a_dist, a_i, a_j))

    # get closest pairs between points and each clusters
    for cluster in recently_created_clusters:
        c = cluster['elements']
        b_i, b_j, b_dist = closest_dist_between_clusters(dataset, c, points)
        if b_i in points:
            pairs.append((b_i, b_j, b_dist, b_i, cluster['id']))
        else:
            pairs.append((b_i, b_j, b_dist, b_j, cluster['id']))

    # get closest pairs among all clusters except points
    for i in range(len(recently_created_clusters)):
        for j in range(i+1, len(recently_created_clusters)):
            c1 = recently_created_clusters[i]
            c2 = recently_created_clusters[j]
            c_i, c_j, c_dist = closest_dist_between_clusters(dataset, c1['elements'], c2['elements'])
            if c1['id'] < c2['id']:
                pairs.append((c_i, c_j, c_dist, c1['id'], c2['id']))
            else:
                pairs.append((c_i, c_j, c_dist, c2['id'], c1['id']))

    min_dist = float('inf')
    for i in pairs:
        if i[2] < min_dist:
            min_dist = i[2]

    min_pairs = []
    for i in pairs:
        if i[2] == min_dist:
            min_pairs.append((i[3], i[4]))

    min_xi = float('inf')
    min_xj = float('inf')
    for pair in min_pairs:
        xi, xj = pair
        if xi < min_xi:
            min_xi = xi
    for pair in min_pairs:
        xi, xj = pair
        if xj < min_xj and xi == min_xi:
            min_xj = xj

    return min_xi, min_xj, min_dist


# helper function for imshow_hac(dataset)
def imshow_closest_dist(dataset, clusters, points):
    recently_created_clusters = []
    for cluster in clusters:
        if cluster['recently_created']:
            recently_created_clusters.append(cluster)

    pairs = []
    # get closest pair among points
    a_i, a_j, a_dist = closest_dist_between_points(dataset, points)
    pairs.append((a_i, a_j, a_dist, a_i, a_j))

    # get closest pairs between points and each clusters
    for cluster in recently_created_clusters:
        c = cluster['elements']
        b_i, b_j, b_dist = closest_dist_between_clusters(dataset, c, points)
        if b_i in points:
            pairs.append((b_i, b_j, b_dist, b_i, cluster['id']))
        else:
            pairs.append((b_i, b_j, b_dist, b_j, cluster['id']))

    # get closest pairs among all clusters except points
    for i in range(len(recently_created_clusters)):
        for j in range(i + 1, len(recently_created_clusters)):
            c1 = recently_created_clusters[i]
            c2 = recently_created_clusters[j]
            c_i, c_j, c_dist = closest_dist_between_clusters(dataset, c1['elements'], c2['elements'])
            if c1['id'] < c2['id']:
                pairs.append((c_i, c_j, c_dist, c1['id'], c2['id']))
            else:
                pairs.append((c_i, c_j, c_dist, c2['id'], c1['id']))

    min_dist = float('inf')
    for i in pairs:
        if i[2] < min_dist:
            min_dist = i[2]

    min_pairs = []
    for i in pairs:
        if i[2] == min_dist:
            min_pairs.append(i)

    min_xi = float('inf')
    for i in min_pairs:
        if i[3] < min_xi:
            min_xi = i[3]

    min_pairs_with_min_xi = []
    for i in min_pairs:
        if i[3] == min_xi:
            min_pairs_with_min_xi.append(i)

    min_xj = float('inf')
    for i in min_pairs_with_min_xi:
        if i[4] < min_xj:
            min_xj = i[4]

    for i in min_pairs_with_min_xi:
        if i[4] == min_xj:
            min_pair_with_min_xi_and_min_xj = i

    p1, p2, dist, xi, xj = min_pair_with_min_xi_and_min_xj

    return p1, p2, dist, xi, xj


# helper function: combine 2 clusters
def combine_clusters(new_id, cluster_1, cluster_2):
    c_id = new_id
    c = []
    c1 = cluster_1['elements']
    c2 = cluster_2['elements']
    for i in c1:
        c.append(i)
    for i in c2:
        c.append(i)
    c = list(dict.fromkeys(c))  # remove duplicates
    count = len(c)              # number of elements
    new_cluster = {'id': c_id, 'elements': c, 'recently_created': True}
    return new_cluster, count


# helper function: combine a cluster and a point
def combine_cluster_and_point(new_id, cluster_1, point):
    c_id = new_id
    c1 = cluster_1['elements']
    c1.append(point)
    c1 = list(dict.fromkeys(c1))  # remove duplicates
    count = len(c1)              # number of elements
    new_cluster = {'id': c_id, 'elements': c1, 'recently_created': True}
    return new_cluster, count


# helper function: get the index of clusters given its id, if not exist, return -1
def get_index(clusters, c_id):
    index = -1
    for i in range(len(clusters)):
        if clusters[i]['id'] == c_id:
            index = i
    return index


def random_x_y(m):
    points = []
    for i in range(m):
        x = random.randint(1, 359)
        y = random.randint(1, 359)
        points.append((x, y))
    return points


def imshow_hac(dataset):
    # pop out invalid data points, i.e. NaN or inf
    filter_dataset = []
    for i in dataset:
        valid = True
        for j in i:
            if math.isinf(j):  # filter inf
                valid = False
            elif math.isnan(j):  # filter NaN
                valid = False
        if valid:
            filter_dataset.append(i)

    # scatter initial status
    x = []
    y = []
    for i in filter_dataset:
        x.append(i[0])
    for j in filter_dataset:
        y.append(j[1])
    plt.scatter(x, y)

    # repeatedly plot linkage
    m = len(filter_dataset)
    points = [i for i in range(m)]
    clusters = []
    x = []
    y = []
    plt.ion()

    for i in range(m - 1):
        plt.pause(0.1)
        p1, p2, dist, xi, xj = imshow_closest_dist(filter_dataset, clusters, points)
        x.clear()
        y.clear()
        x.append(filter_dataset[p1][0])
        x.append(filter_dataset[p2][0])
        y.append(filter_dataset[p1][1])
        y.append(filter_dataset[p2][1])
        if xi <= (m - 1) and xj <= (m - 1):
            points.remove(xi)
            points.remove(xj)
            new_cluster = {'id': m + i, 'elements': [xi, xj], 'recently_created': True}
            clusters.append(new_cluster)
        elif xi <= (m - 1) and xj > (m - 1):
            points.remove(xi)
            index = get_index(clusters, xj)
            clusters[index]['recently_created'] = False
            new_cluster, count = combine_cluster_and_point(m + i, clusters[index], xi)
            clusters.append(new_cluster)
        elif xi > (m - 1) and xj > (m - 1):
            index_i = get_index(clusters, xi)
            index_j = get_index(clusters, xj)
            clusters[index_i]['recently_created'] = False
            clusters[index_j]['recently_created'] = False
            new_cluster, count = combine_clusters(m + i, clusters[index_i], clusters[index_j])
            clusters.append(new_cluster)
        plt.plot(x, y)

    plt.ioff()
    plt.show()

