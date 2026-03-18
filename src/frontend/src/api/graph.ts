const API_URL = import.meta.env.API_URL || "http://localhost:5555";

export async function fetchGraph(nodeId: number) {
  console.log("API URL:", API_URL);

  const res = await fetch(`${API_URL}/graph?node=${nodeId}`);

  if (!res.ok) {
    throw new Error("Failed to fetch graph");
  }

  return res.json();
}

export async function fetchMovie(id: number) {
  const res = await fetch(`${API_URL}/movie?id=${id}`);

  if (!res.ok) {
    throw new Error("Movie not found");
  }

  return res.json();
}