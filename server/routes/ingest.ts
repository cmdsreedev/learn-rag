import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { chunkText, embedText, storeVectors } from '../services';

export default async function ingestRoute(app: FastifyInstance) {
    app.post('/ingest', async (request: FastifyRequest, reply: FastifyReply) => {
        console.log('[ingest] Request received, content-type:', request.headers['content-type']);
        console.log('[ingest] Raw body type:', typeof request.body, '| body:', JSON.stringify(request.body)?.slice(0, 200));

        const body = request.body as { text: string };

        if (!body || typeof body.text !== 'string') {
            console.error('[ingest] Invalid body — missing or non-string text field. Got:', body);
            return reply.status(400).send({ error: 'Missing text field' });
        }

        console.log('[ingest] Text length:', body.text.length);

        const chunks = chunkText(body.text);
        console.log('[ingest] Chunks produced:', chunks.length, '| First chunk preview:', chunks[0]?.slice(0, 80));

        console.log('[ingest] Starting embedding for', chunks.length, 'chunks...');
        const vectors = await Promise.all(chunks.map(embedText));
        console.log('[ingest] Embeddings done. Sample vector length:', vectors[0]?.length);

        storeVectors(chunks, vectors);
        console.log('[ingest] Vectors stored successfully.');

        return reply.status(200).send({ chunks: chunks.length });
    });
}