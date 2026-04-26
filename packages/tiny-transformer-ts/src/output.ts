import type { Matrix } from './common';
import { matMul, softmax } from './utils';

export class OutputLayer {
  W: Matrix; // maps vector → vocab

  constructor(W: Matrix) {
    this.W = W;
  }

  forward(input: Matrix): Matrix {
    const logits = matMul(input, this.W);
    return logits.map((row) => softmax(row));
  }
}