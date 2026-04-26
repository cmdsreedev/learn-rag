import type { Matrix } from './common';

export function transpose(m: Matrix): Matrix {
  if (m.length === 0) return [];
  return m[0].map((_, col) => m.map((row) => row[col]));
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  if (a.length === 0 || b.length === 0) return [];

  return a.map((row) =>
    b[0].map((_, col) => row.reduce((sum, val, i) => sum + val * b[i][col], 0))
  );
}

export function softmax(row: number[]): number[] {
  const max = Math.max(...row);
  const exps = row.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

export function relu(m: Matrix): Matrix {
  return m.map((row) => row.map((x) => Math.max(0, x)));
}

export function addMatrices(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}