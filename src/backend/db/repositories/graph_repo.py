from db.models.embedding import Embedding
from db.models.graph import Graph
from db.models.movie import Movie
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy_utils import Ltree


class GraphRepository:
    """
    Repository for performing db related operations.

    Core idea is to use Ltree for tree-structured graph, since it
    allows fast lookups. For vectors - pgvector is used
    """
    def __init__(self, session: AsyncSession) -> None:
        self.session = session


    async def create_root(self) -> Graph:
        """
        Creates root node

        Returns:
            Graph: root node
        """

        # Check if root already exists
        result = await self.session.execute(
            select(Graph).where(Graph.path == Ltree('root'))
        )
        root = result.scalar_one_or_none()

        if root is not None:
            print('Root already exists')
            return root

        # Create a new one
        root = Graph(
            name='All movies',
            path=Ltree('root'),
            type='node',
        )

        self.session.add(root)
        await self.session.commit()
        await self.session.refresh(root)

        return root


    async def add_child(self, parent_id: int, name: str) -> Graph:
        """
        Adds node to graph

        Args:
            parent_id (int): id of parent's node
            name (str): node name

        Raises:
            RuntimeError: No parent with passed id

        Returns:
            Graph: created node
        """
        parent = await self.session.get(Graph, parent_id)

        if parent is None:
            raise RuntimeError(f'No parent with id {parent_id}')

        child = Graph(
            name=name,
            type='node',
            path=Ltree('tmp'),
        )

        self.session.add(child)
        await self.session.flush()  # gets child.id

        valid_path = f'{parent.path}.{child.id}'
        child.path = Ltree(valid_path)

        parent.children_count += 1

        await self.session.commit()
        await self.session.refresh(child)

        return child


    async def get_immediate_children(
        self, node_id: int
    ) -> dict[str, Graph | list[Graph] | list[Movie]]:
        """
        Gets all the node's children

        Args:
            node_id (int): node's id

        Raises:
            RuntimeError: no node with passed id

        Returns:
            dict[str, Graph | list[Graph] | list[Movie]]: dict with node,
                child nodes, and directly connected movies
        """
        node = await self.session.get(Graph, node_id) # get the node to extract its path

        if node is None:
            raise RuntimeError(f'No node with id {node_id}')

        # Convert Ltree to string for serialization
        if hasattr(node, 'path') and not isinstance(node.path, str):
            node.path = str(node.path)

        # .*{1} selects all the 1 lvl depth children
        query_path = str(node.path) + '.*{1}'

        result = await self.session.execute(
            select(Graph).where(text("path ~ :pattern")).params(pattern=query_path)
        )
        children = result.scalars().all()

        # Check for movies in this node
        result = await self.session.execute(
            select(Movie).where(Movie.graph_id == node_id)
        )
        movies = result.scalars().all()

        return {
            'node': node,
            'children_nodes': children,
            'movies': movies,
        }


    async def add_movie(
        self,
        graph_id: int,
        title: str,
        year: int,
        vectors: list[list[float]],
    ) -> Movie:
        """
        Adds movie to the node

        Args:
            graph_id (int): node's id
            title (str): movie title
            year (int): movie year
            vectors (list[list[float]]): movie's embeddings

        Returns:
            Movie: created movie instance
        """
        movie = Movie(
            graph_id=graph_id,
            title=title,
            year=year,
        )

        self.session.add(movie)
        await self.session.flush()  # get movie.id

        embs = [
            Embedding(
                movie_id=movie.id,
                window_id=i,
                embedding=vec,
            )
            for i, vec in enumerate(vectors)
        ]

        self.session.add_all(embs)

        await self.session.commit()
        await self.session.refresh(movie)

        return movie


    async def get_movie(self, movie_id: int) -> Movie | None:
        """
        Get movie's data and embeddings

        Args:
            movie_id (int): id

        Returns:
            Movie | None: movie with embeddings
        """
        query = (
            select(Movie)
            .options(selectinload(Movie.embeddings))
            .where(Movie.id == movie_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_all_movies(self) -> list[Movie]:
        query = select(Movie).order_by(Movie.id)
        result = await self.session.execute(query)
        return result.scalars().all()