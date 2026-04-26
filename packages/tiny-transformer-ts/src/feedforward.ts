type Matrix = number[][];

export class FeedForward {
  W1: Matrix;
  W2: Matrix;

  constructor(W1: Matrix, W2: Matrix) {
    this.W1 = W1;
    this.W2 = W2;
  }

  matMul(a: Matrix, b: Matrix): Matrix {
    return a.map((row) =>
      b[0].map((_, col) => row.reduce((sum, val, i) => sum + val * b[i][col], 0))
    );
  }

  relu(m: Matrix): Matrix {
    return m.map(row => row.map(x => Math.max(0, x)));
  }

  forward(input: Matrix): Matrix {
    const hidden = this.relu(this.matMul(input, this.W1))
    return this.matMul(hidden, this.W2)
  }
}
