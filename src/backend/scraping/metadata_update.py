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
    last_printed_processed = 0

    total_start = time.perf_counter()
    batch_start = time.perf_counter()

    # Process 20 films concurrently. You can tweak this,
    # but setting it too high might trigger Wikipedia rate limits.
    BATCH_SIZE = 20

    async for session in get_db():
        graph = GraphRepository(session)
        movies = await graph.get_all_movies()
        total_movies = len(movies)

        # Iterate through the movies in chunks of BATCH_SIZE
        for i in range(0, total_movies, BATCH_SIZE):
            batch = movies[i:i + BATCH_SIZE]

            # 1. Create a list of tasks. asyncio.to_thread runs the synchronous
            # get_film_metadata function in a separate thread for each movie.
            tasks = [asyncio.to_thread(get_film_metadata, movie.title) for movie in batch]

            # 2. Run them all concurrently and wait for the batch to finish
            results = await asyncio.gather(*tasks)

            # 3. Pair the original movies back up with their fetched metadata
            for movie, metadata in zip(batch, results):
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

            # Logging logic adjusted for batches (fires roughly every 100 processed)
            if processed - last_printed_processed >= 100:
                batch_time = time.perf_counter() - batch_start
                films_in_batch = processed - last_printed_processed
                avg_per_film = batch_time / films_in_batch
                estimated_total = avg_per_film * total_movies
                elapsed_total = time.perf_counter() - total_start

                print(
                    f"Processed {processed} films | "
                    f"avg per film: {avg_per_film:.2f}s | "
                    f"est. total for {total_movies}: {estimated_total:.2f}s | "
                    f"elapsed: {elapsed_total:.2f}s"
                )

                batch_start = time.perf_counter()
                last_printed_processed = processed

            # Brief pause to respect Wikipedia's servers and avoid IP bans
            await asyncio.sleep(0.5)

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