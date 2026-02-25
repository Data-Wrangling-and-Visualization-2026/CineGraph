import glob
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from db.repositories.graph_repo import GraphRepository
from db.session import SessionLocal
from settings import settings
from sklearn.cluster import KMeans


def random_name() -> str:
    characters = string.ascii_letters + string.digits
    random_string_list = random.choices(characters, k=10)
    random_string = ''.join(random_string_list)
    return random_string


class GraphCreator:
    def __init__(self):
        self.db = SessionLocal()
        self.repo = GraphRepository(self.db)


    def _construct_dataset(self) -> pd.DataFrame:
        movies = glob.glob(settings.emotion_analyzer.output_path + '/*.csv')

        self.emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        emotions_std = ['sadness_std', 'joy_std', 'love_std', 'anger_std', 'fear_std', 'surprise_std']

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


        all_movies_emb = pd.DataFrame(data=all_movies_emb, columns=self.emotions + emotions_std)
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


    def _add_movies_to_node(self, node, movies) -> None:
        titles = movies['movie']
        for title in titles:
            movie_data = self._collect_movie_data(title)
            self.repo.add_movie(node.id, **movie_data)


    def _construct_cluster(self, node, movies_subset: pd.DataFrame, depth):
        print('\t' * depth, len(movies_subset), sep='| ')

        if depth == settings.graph.max_depth:
            self._add_movies_to_node(node, movies_subset)
            return

        clusters = KMeans(
            n_clusters=settings.graph.num_clusters,
            n_init=10,
            random_state=42
        ).fit_predict(movies_subset[self.emotions])

        for idx in range(settings.graph.num_clusters):
            selected = movies_subset[clusters == idx]
            if len(selected) <= settings.graph.min_samples_leaf:
                self._add_movies_to_node(node, selected)
            else:
                new_node = self.repo.add_child(node.id, random_name())
                self._construct_cluster(new_node, selected, depth + 1)


    def construct_graph(self):
        movies = self._construct_dataset()
        root = self.repo.create_root()
        self._construct_cluster(root, movies, 0)
