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


def random_name() -> str:
    """
    Generates random string

    Returns:
        str: _description_
    """
    characters = string.ascii_letters + string.digits
    random_string_list = random.choices(characters, k=10)
    random_string = ''.join(random_string_list)
    return random_string


class GraphCreator:
    """
    Performs clustering and saves the data to DB.

    Clustering algorithm - Hierarchical clustering based on KMeans + agglomerative clustering:
        1. All movies are divided into desired number of node using KMeans (mini batch version)
        2. Created clusters are united into tree using agglomerative clustering
        3. After the full tree is created, balancing algorithm is used
            - Leaf's movies can reconnected to its parent if too few of them
            - Node can be merged into its parent, if their centroids are too close
        4. This tree structure is saved into DB with corresponding emotion embeddings
    """
    def __init__(self, include_std: bool = True, num_acts: int = 3, delta_threshold: float = 0.2):
        """
        Args:
            include_std (bool, optional): Include/exclude standard deviation from
                                          features for clustering. Defaults to True.
            num_acts (int, optional): Number of intervals over which features are
                                      calculated. Defaults to 3.
            delta_threshold (float, optional): Threshold after which feature is considered
                                               as dominant. Defaults to 0.2.
        """
        self.num_acts = num_acts
        self.include_std = include_std
        self.delta_threshold = delta_threshold
        self.emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        # self.emotions_std = [emotion + '_std' for emotion in self.emotions]

        self.features = []

        # Each act is simple movie[i: i + act_length]
        for act in range(num_acts):
            self.features.extend(
                [emotion + f'_act{act+1}' for emotion in self.emotions]
            )

        if include_std:
            self.features.extend([emotion + '_std' for emotion in self.emotions])


    def _construct_dataset(self) -> pd.DataFrame:
        """
        Selects all the files and combines them into two datasets:
            1. Raw features (for each act)
            2. Scaled version (via StandardScaler)

        Returns:
            pd.DataFrame: raw features with movies' names
        """
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

            # Split each movie into `num_acts` intervals
            acts = np.array_split(emb[self.emotions].values, self.num_acts)

            # Calculate the `centroid` for each interval
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

        # Scaling will be helpful later in distance calculation
        self.scaled_features = StandardScaler().fit_transform(self.all_movies_emb[self.features])
        return all_movies_emb


    def _construct_emotional_shift(self, child_centroid: np.ndarray, parent_centroid: np.ndarray) -> str:
        """
        Selects three dominant emotions:
            1. Difference between two centroids is calculated_type_
            2. Two emotions which are higher in child are selected
            3. One emotion which is higher in parent is selected
        If all the diffs are too small base string is returned

        Used for better LLM context to produce more diverse names

        Args:
            child_centroid (ndarray): centroid
            parent_centroid (ndarray): centroid

        Returns:
            str: formatted string with emotional shifts description
        """
        if parent_centroid is None: # Root node does not have any centroid
            return 'Baseline Story Shape'

        # For simplicity we exclude standard deviation from consideration
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


    def _build_hierarchy(self) -> dict:
        """
        Creates tree structure based on clusters

        Output tree in each node/leaf has:
            type: (node/root/leaf)
            indices: indices of attached movies
            count: number of attached movies
            children: attached sub-nodes
        Additionally, each node has distance - distance from parent centroid

        Returns:
            dict: tree as dictionary
        """
        movies = self._construct_dataset()
        n = len(movies)

        # Number of initial clusters
        n_micro = min(settings.graph.max_nodes, max(100, n // settings.graph.target_leaf_size))

        # Clusters creation
        kmeans = MiniBatchKMeans(n_clusters=n_micro, batch_size=2048, random_state=42)
        labels = kmeans.fit_predict(self.scaled_features)

        # Create mapping [cluster_id, attached movies (ids)]
        clusters_mapping = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_mapping[label].append(idx)

        # Agglomerative clustering
        centroids = kmeans.cluster_centers_
        Z = linkage(centroids, method='ward')
        root, _ = to_tree(Z, rd=True)

        # Convert to dict
        tree = self._convert_tree(root, clusters_mapping)
        with open('./clustering/unbalanced.json', 'w') as f:
            json.dump(tree, f, indent=4)

        # Rebalance
        tree = self._rebalance_tree(tree)
        with open('./clustering/balanced.json', 'w') as f:
            json.dump(tree, f, indent=4)

        return tree


    def _convert_tree(self, node: ClusterNode, centroids_mapping: dict[int, list[int]]) -> dict:
        """
        Convert tree to dict

        Args:
            node (ClusterNode): node
            centroids_mapping (dict[int, list[int]]): mapping of clusters and movies

        Returns:
            dict: tree
        """
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
        """
        Rebalance the tree:
            - Leaf's movies can reconnected to its parent if too few of them
            - Node can be merged into its parent, if their centroids are too close

        Args:
            node (dict): node
            depth (int, optional): current node's depth. Defaults to 0.

        Returns:
            dict: rebalanced tree
        """
        if not node.get('children') or depth >= settings.graph.max_depth:
            node['type'] = 'leaf'
            return node

        # Recursively rebalance all the nodes
        node['children'] = [self._rebalance_tree(child, depth+1) for child in node['children']]

        # Stopping criteria - nothing has changed after the last iteration
        changed = True
        while changed and len(node['children']) < settings.graph.max_fanout:
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


    def _collect_movie_data(self, title: str) -> dict:
        """
        Collects movie related data

        Args:
            title (str): movies title (with year)

        Returns:
            dict: title, year, vectors (embeddings as df)
        """
        path = settings.emotion_analyzer.output_path + f"/{title.replace(' ', '_')}.csv"
        df = pd.read_csv(path)
        embeddings = df[self.emotions].values

        return {
            'title': ''.join(title.split()[:-1]),
            'year': int(title.split()[-1]),
            'vectors': embeddings
        }


    async def _add_movies_to_node(self, node, movies) -> None:
        """
        Attaches movies to node

        Args:
            node (Graph): DB node
            movies (dict): movies
        """
        titles = movies['movie']
        for title in titles:
            movie_data = self._collect_movie_data(title)
            await self.repo.add_movie(node.id, **movie_data)


    async def _populate_db_from_tree(self, node, parent_db_node, parent_centroid = None, indent: int = 0) -> None:
        """
        Dumps tree into db

        Args:
            node (Graph): DB node
            parent_db_node (Graph): DB node
            parent_centroid (np.ndarray, optional): centroid. Defaults to None.
            indent (int, optional): indent of node for logging. Defaults to 0.
        """
        children = node.get('children', [])
        if not children: # If not children -> leaf with movies only
            await self._add_movies_to_node(
                parent_db_node,
                self.all_movies_emb.iloc[node['indices']]
            )
            return

        groups = []
        child_centroids = []

        # Select the most `representative` movies for each group
        # to pass to LLM for naming. We can not pass all the movies
        # due to context length restriction
        for child in children:
            indices = child['indices']
            child_vectors = self.scaled_features[indices]
            child_centroid = child_vectors.mean(axis=0)
            child_centroids.append(child_centroid)

            # Select only the closest to parent centroid
            distances = np.linalg.norm(child_vectors - child_centroid, axis=1)
            closest = np.argsort(distances)[:15]

            selected = [indices[i] for i in closest]
            titles = self.all_movies_emb.iloc[selected]['movie'].values

            shift = self._construct_emotional_shift(child_centroid, parent_centroid)

            groups.append({
                'titles': titles,
                'shift': shift,
            })

        # Generate names
        names = generate_context_aware_node_name(parent_db_node.name, groups)

        # Dump into db
        for idx, child in enumerate(children):
            node_name = names[idx]
            child_node = await self.repo.add_child(
                parent_id=parent_db_node.id,
                name=node_name,
                centroid=child_centroids[idx]
            )

            print('=>' * indent, node_name, child['count'])

            await self._populate_db_from_tree(
                node=child,
                parent_db_node=child_node,
                parent_centroid=child_centroids[idx],
                indent=indent+1
            )


    async def construct_graph(self):
        """
        Constructs graph and saves to db
        """
        async for db in get_db():
            self.repo = GraphRepository(db)

            tree = self._build_hierarchy()
            root_centroid = self.scaled_features.mean(axis=0)

            root = await self.repo.create_root(centroid=root_centroid)
            await self._populate_db_from_tree(tree, root, root_centroid)
