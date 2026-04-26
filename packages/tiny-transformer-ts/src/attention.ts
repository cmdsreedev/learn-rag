type Matrix = number[][];

export class Attention {
  Wq: Matrix;
  Wk: Matrix;
  Wv: Matrix;

  constructor(Wq: Matrix, Wk: Matrix, Wv: Matrix) {
    this.Wq = Wq;
    this.Wk = Wk;
    this.Wv = Wv;
  }

  transpose(m: Matrix): Matrix {
    return m[0].map((_, col) => m.map(row => row[col]));
  }

  matMul(a: Matrix, b: Matrix): Matrix {
    return a.map(row =>
      b[0].map((_, col) =>
        row.reduce((sum, val, i) => sum + val * b[i][col], 0)
      )
    );
  }

  softmax(row: number[]): number[] {
    const max = Math.max(...row);
    const exps = row.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  forward(input: Matrix): Matrix {
    // 1. create views
    const Q = this.matMul(input, this.Wq);
    const K = this.matMul(input, this.Wk);
    const V = this.matMul(input, this.Wv);

    // 2. compare
    const scores = this.matMul(Q, this.transpose(K));

    // 3. importance
    const weights = scores.map(row => this.softmax(row));

    // 4. mix
    const output = this.matMul(weights, V);

    return output;
  }
}