import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { embedText, searchVectors, generateAnswer } from '../services';

export default async function (app: FastifyInstance) {
    app.post('/query',async (request: FastifyRequest, reply: FastifyReply) => {
        const { question } = request.body as  { question: string}
        console.log('[query] Question:', question);

        const qVec = await embedText(question);
        console.log('[query] Question vector length:', qVec?.length);

        const docs = await searchVectors(qVec, 5);
        console.log('[query] Docs retrieved:', docs.length, '| First doc preview:', docs[0]?.slice(0, 80));

        const answer = await generateAnswer(question, docs);
        console.log('[query] Answer received:', answer?.slice(0, 200));

        return { answer, docs };
    })
}