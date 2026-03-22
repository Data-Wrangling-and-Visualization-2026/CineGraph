import type { GraphData, MyNode, MyLink } from "../types/graph";

export function transformGraphData(apiData: any): GraphData {
  const nodes: MyNode[] = [];
  const links: MyLink[] = [];
  const root = apiData;

  // Root
  nodes.push({ id: String(root.id), name: root.name, group: 1, val: 15, level: 1 });

  // Children
  root.children_nodes?.forEach((child: any) => {
    nodes.push({
      id: String(child.id),
      name: child.name,
      group: 2,
      // Ensure a minimum size of 8, max out at 20 for visual clarity
      val: Math.min(Math.max((child.children_count || 1) * 2, 8), 20),
      level: 2
    });
    links.push({ source: String(root.id), target: String(child.id) });
  });

  // Movies
  root.movies?.forEach((movie: any) => {
    nodes.push({
      id: `movie-${movie.id}`,
      name: movie.title,
      group: 3,
      val: 6,
      level: 3 // Make sure this is definitely 3
    });
    links.push({ source: String(root.id), target: `movie-${movie.id}` });
  });

  return { nodes, links };
}