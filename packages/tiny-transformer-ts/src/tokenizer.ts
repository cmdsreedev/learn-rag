type Vocab = Record<string, number>;

export class Tokenizer {
  vocab: Vocab = {};
  reverseVocab: Record<number, string> = {};
  nextId = 0;

  fit(text: string) {
    const tokens = this.tokenize(text);

    for (const token of tokens) {
      if (!(token in this.vocab)) {
        this.vocab[token] = this.nextId;
        this.reverseVocab[this.nextId] = token;
        this.nextId++;
      }
    }
  }

  tokenize(text: string): string[] {
    return text.toLowerCase().split(/\s+/);
  }

  encode(text: string): number[] {
    const tokens = this.tokenize(text);
    return tokens.map((token) => this.vocab[token]);
  }

  decode(ids: number[]): string {
    return ids.map((id) => this.reverseVocab[id]).join(' ');
  }
}
