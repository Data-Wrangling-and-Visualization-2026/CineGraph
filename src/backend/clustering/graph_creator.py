import glob
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from clustering.utils import generate_context_aware_node_name
from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from settings import settings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def random_name() -> str:
    characters = string.ascii_letters + string.digits
    random_string_list = random.choices(characters, k=10)
    random_string = ''.join(random_string_list)
    return random_string


class GraphCreator:

    def _construct_dataset(self) -> pd.DataFrame:
        movies = glob.glob(settings.emotion_analyzer.output_path + '/*.csv')

        self.emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.emotions_std = ['sadness_std', 'joy_std', 'love_std', 'anger_std', 'fear_std', 'surprise_std']
        self.features = self.emotions + self.emotions_std

        all_movies_emb = []
        names = []

        for _, movie in enumerate(movies):
            names.append(
                Path(movie).stem.replace('_', ' ')
            )

            emb = pd.read_csv(movie)

            avg_pooling = emb[self.emotions].mean(axis=0).values
            std_pooling = emb[self.emotions].std(axis=0).values
            clst_emb = np.concat([avg_pooling, std_pooling])

            all_movies_emb.append(clst_emb)

        all_movies_emb = pd.DataFrame(data=all_movies_emb, columns=self.features)
        all_movies_emb['movie'] = names

        return all_movies_emb


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


    async def _construct_cluster(self, node, movies_subset: pd.DataFrame, depth: int, name: str = 'Root'):
        print('\t' * depth, name, len(movies_subset), sep='| ')

        if depth == settings.graph.max_depth:
            await self._add_movies_to_node(node, movies_subset)
            return

        scaled_features = StandardScaler().fit_transform(movies_subset[self.features])
        clusters = KMeans(
            n_clusters=settings.graph.num_clusters,
            n_init=10,
            random_state=42
        ).fit_predict(scaled_features)


        if len(movies_subset) < 1001:
            groups = []
            for idx in range(settings.graph.num_clusters):
                selected = movies_subset[clusters == idx]['movie'].values
                groups.append(
                    selected
                )

            names = generate_context_aware_node_name(groups)
        else:
            names = [random_name() for _ in range(settings.graph.num_clusters)]


        for idx in range(settings.graph.num_clusters):
            selected = movies_subset[clusters == idx]
            if len(selected) <= settings.graph.min_samples_leaf:
                await self._add_movies_to_node(node, selected)
            else:
                new_node = await self.repo.add_child(node.id, names[idx])
                await self._construct_cluster(new_node, selected, depth + 1, names[idx])


    async def construct_graph(self):
        async for db in get_db():
            self.repo = GraphRepository(db)

            movies = self._construct_dataset()
            root = await self.repo.create_root()
            await self._construct_cluster(root, movies, 0)
