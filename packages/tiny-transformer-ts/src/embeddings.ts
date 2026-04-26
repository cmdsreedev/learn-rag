import type { Matrix } from './common';

export class Embedding {
    vocabSize: number;
    dim: number;
    table: Matrix;

    constructor(vocabSize: number, dim:number){
        this.vocabSize = vocabSize
        this.dim = dim

        this.table = Array.from({ length: vocabSize}, () => 
            Array.from({ length: dim }, () => Math.random() * 2 -1)
        )
    }

    forward(ids: number[]): Matrix{
        return ids.map(id => this.table[id])
    }
}