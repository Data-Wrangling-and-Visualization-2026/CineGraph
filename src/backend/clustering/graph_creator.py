import glob
import json
import random
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from clustering.utils import generate_context_aware_node_name
from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from settings import settings
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

MAX_NODES = 800
TARGET_LEAF_SIZE = 50
MIN_FANOUT = 3
MAX_FANOUT = 8
MAX_DEPTH = 5


def random_name() -> str:
    characters = string.ascii_letters + string.digits
    random_string_list = random.choices(characters, k=10)
    random_string = ''.join(random_string_list)
    return random_string


class GraphCreator:

    def __init__(self, include_std: bool = True, num_acts: int = 3, delta_threshold: float = 0.2):
        self.num_acts = num_acts
        self.include_std = include_std
        self.delta_threshold = delta_threshold
        self.emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        # self.emotions_std = [emotion + '_std' for emotion in self.emotions]

        self.features = []
        for act in range(num_acts):
            self.features.extend(
                [emotion + f'_act{act+1}' for emotion in self.emotions]
            )

        if include_std:
            self.features.extend([emotion + '_std' for emotion in self.emotions])


    def _construct_dataset(self) -> pd.DataFrame:
        movies = glob.glob(settings.emotion_analyzer.output_path + '/*.csv')

        all_movies_emb = []
        names = []

        for _, movie in enumerate(movies):
            emb = pd.read_csv(movie)
            if len(emb) < self.num_acts:
                continue

            names.append(
                Path(movie).stem.replace('_', ' ')
            )

            acts = np.array_split(emb[self.emotions].values, self.num_acts)

            data = [
                acts[i].mean(axis=0) for i in range(self.num_acts)
            ]
            if self.include_std:
                data.append(
                    emb[self.emotions].std(axis=0).values
                )

            clst_emb = np.concat(data)
            all_movies_emb.append(clst_emb)

        self.all_movies_emb = pd.DataFrame(data=all_movies_emb, columns=self.features)
        self.all_movies_emb['movie'] = names

        self.scaled_features = StandardScaler().fit_transform(self.all_movies_emb[self.features])
        return all_movies_emb


    def _construct_emotional_shift(self, child_centroid, parent_centroid) -> str:
        if parent_centroid is None:
            return 'Baseline Story Shape'

        deltas = child_centroid - parent_centroid
        if self.include_std:
            deltas = deltas[:-len(self.emotions)]

        shifts = []

        sorted_idx = np.argsort(deltas)

        for idx in sorted_idx[-2:]:
            if deltas[idx] > self.delta_threshold:
                shifts.append(
                    f"Higher {self.features[idx].replace('_', ' in ')}"
                )

        for idx in sorted_idx[:1]:
            if deltas[idx] < -self.delta_threshold:
                shifts.append(
                    f"Lower {self.features[idx].replace('_', ' in ')}"
                )

        return ', '.join(shifts) if shifts else 'Balanced/Nuanced Pacing'


    def _build_hierarchy(self):
        movies = self._construct_dataset()
        n = len(movies)

        n_micro = min(MAX_NODES, max(100, n // TARGET_LEAF_SIZE))
        kmeans = MiniBatchKMeans(n_clusters=n_micro, batch_size=2048, random_state=42)
        labels = kmeans.fit_predict(self.scaled_features)

        clusters_mapping = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_mapping[label].append(idx)

        centroids = kmeans.cluster_centers_
        Z = linkage(centroids, method='ward')
        root, _ = to_tree(Z, rd=True)

        tree = self._convert_tree(root, clusters_mapping)
        with open('./clustering/unbalanced.json', 'w') as f:
            json.dump(tree, f, indent=4)

        tree = self._rebalance_tree(tree)
        with open('./clustering/balanced.json', 'w') as f:
            json.dump(tree, f, indent=4)

        return tree


    def _convert_tree(self, node: ClusterNode, centroids_mapping: dict[int, list[int]]) -> dict:
        if node.is_leaf():
            members = centroids_mapping[node.id]
            return {'type': 'leaf', 'indices': members, 'count': len(members), 'children': []}

        left = self._convert_tree(node.left, centroids_mapping)
        right = self._convert_tree(node.right, centroids_mapping)
        combined_indices = left.get('indices', []) + right.get('indices', [])

        return {
            'type': 'node',
            'distance': node.dist,
            'count': left['count'] + right['count'],
            'indices': combined_indices,
            'children': [left, right]
        }


    def _rebalance_tree(self, node: dict, depth: int = 0) -> dict:
        if not node.get('children') or depth >= MAX_DEPTH:
            node['type'] = 'leaf'
            return node

        node['children'] = [self._rebalance_tree(child, depth+1) for child in node['children']]

        changed = True
        while changed and len(node['children']) < MAX_FANOUT:
            changed = False
            new_children = []
            for child in node['children']:
                added = False

                if child is None:
                    print(node)
                    exit(1)

                if child['type'] == 'node':
                    divergence = child.get('distance', 0) / (node.get('distance', 1) + 1e-9)
                    if divergence > 0.65:
                        added = True
                        new_children.extend(child['children'])
                        changed = True

                if not added:
                    new_children.append(child)

            node['children'] = new_children

        return node


    def _collect_movie_data(self, title) -> dict:
        path = settings.emotion_analyzer.output_path + f"/{title.replace(' ', '_')}.csv"
        df = pd.read_csv(path)
        embeddings = df[self.emotions].values

        return {
            'title': ''.join(title.split()[:-1]),
            'year': int(title.split()[-1]),
            'vectors': embeddings
        }


    async def _add_movies_to_node(self, node, movies) -> None:
        titles = movies['movie']
        for title in titles:
            movie_data = self._collect_movie_data(title)
            await self.repo.add_movie(node.id, **movie_data)


    async def _populate_db_from_tree(self, node, parent_db_node, parent_centroid = None, indent: int = 0):
        children = node.get('children', [])
        if not children:
            await self._add_movies_to_node(
                parent_db_node,
                self.all_movies_emb.iloc[node['indices']]
            )
            return

        groups = []
        child_centroids = []

        for child in children:
            indices = child['indices']
            child_vectors = self.scaled_features[indices]
            child_centroid = child_vectors.mean(axis=0)
            child_centroids.append(child_centroid)

            distances = np.linalg.norm(child_vectors - child_centroid, axis=1)
            closest = np.argsort(distances)[:15]

            selected = [indices[i] for i in closest]
            titles = self.all_movies_emb.iloc[selected]['movie'].values

            shift = self._construct_emotional_shift(child_centroid, parent_centroid)

            groups.append({
                'titles': titles,
                'shift': shift,
            })

        names = generate_context_aware_node_name(parent_db_node.name, groups)

        for idx, child in enumerate(children):
            node_name = names[idx]
            child_node = await self.repo.add_child(parent_db_node.id, node_name)

            print('=>' * indent, node_name, child['count'])

            await self._populate_db_from_tree(
                node=child,
                parent_db_node=child_node,
                parent_centroid=child_centroids[idx],
                indent=indent+1
            )


    async def construct_graph(self):
        async for db in get_db():
            self.repo = GraphRepository(db)

            tree = self._build_hierarchy()
            root_centroid = self.scaled_features.mean(axis=0)

            root = await self.repo.create_root()
            await self._populate_db_from_tree(tree, root, root_centroid)
