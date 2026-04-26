const DB: { text: string; vector: number[] }[] = [];

export function storeVectors(texts: string[], vectors: number[][]) {
  for (let i = 0; i < texts.length; i++) {
    DB.push({ text: texts[i] as string, vector: vectors[i] as number[] });
  }
}

function cosine(a: number[], b: number[]) {
  const dot = a.reduce((sum, v, i) => sum + v * (b[i] ?? 0), 0);
  const magA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
  const magB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));

  return dot / (magA * magB);
}

export async function searchVectors(query: number[], k = 5) {
  return DB.map((d) => ({ text: d.text, score: cosine(query, d.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((d) => d.text);
}
