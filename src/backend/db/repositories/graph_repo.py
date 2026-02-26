from db.models.embedding import Embedding
from db.models.graph import Graph
from db.models.movie import Movie

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy_utils import Ltree


class GraphRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_root(self) -> Graph:
        result = await self.session.execute(
            select(Graph).where(Graph.path == Ltree("root"))
        )
        root = result.scalar_one_or_none()

        if root is not None:
            print("Root already exists")
            return root

        root = Graph(
            name="root",
            path=Ltree("root"),
            type="node",
        )

        self.session.add(root)
        await self.session.commit()
        await self.session.refresh(root)

        return root

    async def add_child(self, parent_id: int, name: str) -> Graph:
        parent = await self.session.get(Graph, parent_id)

        if parent is None:
            raise RuntimeError(f"No parent with id {parent_id}")

        child = Graph(
            name=name,
            type="node",
            path=Ltree("tmp"),
        )

        self.session.add(child)
        await self.session.flush()  # gets child.id

        valid_path = f"{parent.path}.{child.id}"
        child.path = Ltree(valid_path)

        parent.children_count += 1

        await self.session.commit()
        await self.session.refresh(child)

        return child

    async def get_immediate_children(
        self, node_id: int
    ) -> dict[str, Graph | list[Graph] | list[Movie]]:
        node = await self.session.get(Graph, node_id)

        if node is None:
            raise RuntimeError(f"No node with id {node_id}")

        query_path = str(node.path) + ".*{1}"

        result = await self.session.execute(
            select(Graph).where(
                Graph.path.lquery(Ltree(query_path))
            )
        )
        children = result.scalars().all()

        result = await self.session.execute(
            select(Movie).where(Movie.graph_id == node_id)
        )
        movies = result.scalars().all()

        return {
            "node": node,
            "children_nodes": children,
            "movies": movies,
        }

    async def add_movie(
        self,
        graph_id: int,
        title: str,
        year: int,
        vectors: list[list[float]],
    ) -> Movie:
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