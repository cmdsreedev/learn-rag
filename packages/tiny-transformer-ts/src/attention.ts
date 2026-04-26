import type { Matrix } from './common';
import { matMul, softmax, transpose } from './utils';

export class Attention {
  Wq: Matrix;
  Wk: Matrix;
  Wv: Matrix;

  constructor(Wq: Matrix, Wk: Matrix, Wv: Matrix) {
    this.Wq = Wq;
    this.Wk = Wk;
    this.Wv = Wv;
  }

  forward(input: Matrix): Matrix {
    // 1. create views
    const Q = matMul(input, this.Wq);
    const K = matMul(input, this.Wk);
    const V = matMul(input, this.Wv);

    // 2. compare
    const scores = matMul(Q, transpose(K));

    // 3. importance
    const weights = scores.map((row) => softmax(row));

    // 4. mix
    const output = matMul(weights, V);

    return output;
  }
}