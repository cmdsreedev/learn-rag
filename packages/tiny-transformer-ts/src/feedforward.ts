import type { Matrix } from './common';
import { matMul, relu } from './utils';

export class FeedForward {
  W1: Matrix;
  W2: Matrix;

  constructor(W1: Matrix, W2: Matrix) {
    this.W1 = W1;
    this.W2 = W2;
  }

  forward(input: Matrix): Matrix {
    const hidden = relu(matMul(input, this.W1));
    return matMul(hidden, this.W2);
  }
}
