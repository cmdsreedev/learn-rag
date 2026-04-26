import { Attention } from './attention';
import { FeedForward } from './feedforward';
import type { Matrix } from './common';
import { addMatrices } from './utils';

export class TransformerBlock {
  attention: Attention;
  feedforward: FeedForward;

  constructor(attention: Attention, feedforward: FeedForward) {
    this.attention = attention;
    this.feedforward = feedforward;
  }

  forward(input: Matrix) {
    const attnOut = this.attention.forward(input);

    const afterAttn = addMatrices(input, attnOut);

    const ffOut = this.feedforward.forward(afterAttn);

    const output = addMatrices(afterAttn, ffOut);

    return output;
  }
}
