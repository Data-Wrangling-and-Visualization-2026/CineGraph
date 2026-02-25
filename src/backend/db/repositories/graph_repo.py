from db.models.embedding import Embedding
from db.models.graph import Graph
from db.models.movie import Movie
from sqlalchemy.orm import Session
from sqlalchemy_utils import Ltree


class GraphRepository:
    def __init__(self, session: Session) -> None:
        self.session = session


    def create_root(self) -> Graph:
        root = self.session.query(Graph).filter(Graph.path == Ltree('root')).first()

        if root is not None:
            print('Root already exists')
            return root

        root = Graph(
            name='root',
            path=Ltree('root'),
            type='node'
        )

        self.session.add(root)
        self.session.commit()

        return root


    def add_child(self, parent_id: int, name: str) -> Graph:
        parent = self.session.query(Graph).get(parent_id)

        if parent is None:
            raise RuntimeError(f'No parent with id {parent_id}')

        child = Graph(
            name=name,
            type='node',
            path=Ltree('tmp')
        )

        self.session.add(child)
        self.session.flush()

        valid_path = f'{parent.path}.{child.id}'
        child.path = Ltree(valid_path)

        parent.children_count += 1

        self.session.commit()

        return child


    def get_immediate_children(self, node_id: int) -> dict[str, Graph | str]:
        node = self.session.query(Graph).get(node_id)
        if node is None:
            raise RuntimeError(f'No node with id {node_id}')

        query_path = str(node.path) + '.*{1}'

        children = self.session.query(Graph).filter(
            Graph.path.lquery(Ltree(query_path))
        ).all()

        movies = self.session.query(Movie).filter(
            Movie.graph_id == node_id
        ).all()

        return {
            'node': node,
            'children_nodes': children,
            'movies': movies
        }


    def add_movie(self, graph_id: int, title: str, year: int, vectors: list[list[float]]) -> Movie:
        movie = Movie(
            graph_id=graph_id,
            title=title,
            year=year,
        )

        self.session.add(movie)
        self.session.flush()

        embs = []
        for i, vec in enumerate(vectors):
            embs.append(
                Embedding(
                    movie_id=movie.id,
                    window_id=i,
                    embedding=vec
                )
            )

        self.session.add_all(embs)
        self.session.commit()

        return movie
