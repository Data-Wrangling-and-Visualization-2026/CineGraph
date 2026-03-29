import asyncio
import json
import time

from db.repositories.graph_repo import GraphRepository
from db.session import get_db
from metadata_parsing import get_film_metadata


async def main():
    updates = []
    processed = 0
    skipped = 0
    batch_start = time.perf_counter()
    total_start = time.perf_counter()

    async for session in get_db():
        graph = GraphRepository(session)
        movies = await graph.get_all_movies()
        total_movies = len(movies)

        for i, movie in enumerate(movies, start=1):

            metadata = get_film_metadata(movie.title)
            if not metadata or metadata.get("error"):
                skipped += 1
                continue

            metadata_json = json.dumps(metadata, ensure_ascii=False)
            escaped_json = metadata_json.replace("'", "''")

            updates.append(
                f"UPDATE public.movies "
                f"SET other_data = '{escaped_json}'::jsonb "
                f"WHERE id = {movie.id};"
            )

            processed += 1

            if processed % 100 == 0:
                batch_time = time.perf_counter() - batch_start
                avg_per_film = batch_time / 100
                estimated_total = avg_per_film * total_movies
                elapsed_total = time.perf_counter() - total_start

                print(
                    f"Processed {processed} films | "
                    f"avg per film: {avg_per_film:.2f}s | "
                    f"est. total for {total_movies}: {estimated_total:.2f}s | "
                    f"elapsed: {elapsed_total:.2f}s"
                )
                batch_start = time.perf_counter()

    total_elapsed = time.perf_counter() - total_start
    print(
        f"Done. processed={processed}, skipped={skipped}, "
        f"total_elapsed={total_elapsed:.2f}s"
    )

    output_path = "/app/generated_sql/02_update_movies_metadata.sql"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("-- Auto-generated metadata update script\n")
        f.write("BEGIN;\n\n")
        for stmt in updates:
            f.write(stmt + "\n")
        f.write("\nCOMMIT;\n")


if __name__ == "__main__":
    asyncio.run(main())