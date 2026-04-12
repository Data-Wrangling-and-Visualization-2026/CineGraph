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
        if depth >= settings.graph.max_depth or not node.get('children'):
            node['type'] = 'leaf'
            node['children'] = []
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

        # Should prevent additional leaf nodes with small # of movies
        absorbed, remaining = [], []
        for child in node['children']:
            if child['count'] < settings.graph.min_samples_leaf and len(child['children']) > 1:
                absorbed.extend(child['indices'])
            else:
                remaining.append(child)

        if absorbed and remaining:
            centroids = [
                self.scaled_features[
                    c['indices']
                ].mean(axis=0) for c in remaining
            ]
            for idx in absorbed: # find the best node to attach movie
                dists = [np.linalg.norm(self.scaled_features[idx] - c) for c in centroids]
                best = int(np.argmin(dists))
                remaining[best]['indices'].append(idx)
                remaining[best]['count'] += 1
        elif absorbed:
            node['type'] = 'leaf'
            node['children'] = []
            return node

        node['children'] = remaining

        # Should prevent long chains of one node
        if len(node['children']) == 1 and len(node['indices']) < settings.graph.min_samples_leaf:
            only = node['children'][0]
            node['indices'] = only['indices']
            node['count'] = only['count']
            node['children'] = only.get('children', [])
            if not node['children']:
                node['type'] = 'leaf'

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

        emotion_arc = np.concat(
            [
                act.mean(axis=0)
                for act in np.array_split(
                    embeddings,
                    indices_or_sections=3,
                    axis=0
                )
            ] + [embeddings.std(axis=0)],
        )

        return {
            'title': ' '.join(title.split()[:-1]), # [:-1] removes year
            'year': int(title.split()[-1]),
            'emotion_arc': emotion_arc,
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


    def _split_oversized_leaves(self, node: dict, max_size: int = 150):
        """
        Recursively finds leaves with too many movies and splits them using KMeans.
        """
        if node['type'] == 'leaf':
            if node['count'] > max_size:
                # Calculate how many clusters we need to get sizes roughly around 50-70
                k = max(2, node['count'] // (max_size // 2))

                vectors = self.scaled_features[node['indices']]
                kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42)
                labels = kmeans.fit_predict(vectors)
                indices = np.array(node['indices'])

                new_children = []
                for i in range(k):
                    # Map the local cluster indices back to the global dataset indices
                    local_indices = indices[labels == i]

                    child_centroid = self.scaled_features[local_indices].mean(axis=0)
                    new_children.append({
                        'type': 'leaf',
                        'indices': local_indices.tolist(),
                        'count': len(local_indices),
                        'centroid': child_centroid,
                        'children': []
                    })

                # Transform this leaf into a parent node
                node['type'] = 'node'
                node['children'] = new_children


        elif node.get('children'):
            for child in node['children']:
                self._split_oversized_leaves(child, max_size)


    def _assign_names(self, node: dict, depth: int = 0) -> str:
        children = node.get('children', [])
        if not children:
            indices = node['indices']

            node_vectors = self.scaled_features[indices]
            node_centroid = node_vectors.mean(axis=0)

            # Select only the closest to parent centroid
            distances = np.linalg.norm(node_vectors - node_centroid, axis=1)
            closest = np.argsort(distances)[:10]

            selected = [indices[i] for i in closest]
            titles = self.all_movies_emb.iloc[selected]['movie'].values

            node['name'] = generate_context_aware_node_name(titles, leaf=True)
            print('\t' * depth + node['name'])
            return node['name']

        child_names = [self._assign_names(child, depth+1) for child in children]

        if depth == 0:
            return

        node['name'] = generate_context_aware_node_name(child_names[:10], leaf=False)
        print('\t' * depth + node['name'])

        return node['name']


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

        child_centroids = []

        # Select the most `representative` movies for each group
        # to pass to LLM for naming. We can not pass all the movies
        # due to context length restriction
        for child in children:
            indices = child['indices']
            child_vectors = self.scaled_features[indices]
            child_centroid = child_vectors.mean(axis=0)
            child_centroids.append(child_centroid)

        # Dump into db
        for idx, child in enumerate(children):
            child_node = await self.repo.add_child(
                parent_id=parent_db_node.id,
                name=child['name'],
                centroid=child_centroids[idx]
            )

            print('=>' * indent, child['name'], child['count'])

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
            self._split_oversized_leaves(tree, max_size=settings.graph.target_leaf_size)

            self._assign_names(tree, depth=0)

            root_centroid = self.scaled_features.mean(axis=0)

            root = await self.repo.create_root(centroid=root_centroid)
            await self._populate_db_from_tree(tree, root, root_centroid)
