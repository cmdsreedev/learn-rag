import Fastify from 'fastify';
import ingestRoute from './routes/ingest';
import queryRoute from './routes/query';
import process from 'process';

const app = Fastify({ logger: true });
app.register(ingestRoute);
app.register(queryRoute);

let shuttingDown = false;

async function shutdown(signal: string) {
  if (shuttingDown) return;
  shuttingDown = true;

  app.log.info(`${signal} received, closing server gracefully`);

  // If graceful shutdown hangs on open sockets, force exit.
  const forceExit = setTimeout(() => {
    app.log.error('Graceful shutdown timed out, forcing exit');
    process.exit(1);
  }, 5000);
  forceExit.unref();

  try {
    await app.close();
    process.exit(0);
  } catch (err) {
    app.log.error(err, 'Error during server shutdown');
    process.exit(1);
  }
}

app.listen({ port: 3000 }, (err, address) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
  app.log.info(`Server listening at ${address}`);
});

process.on('SIGINT', () => {
  void shutdown('SIGINT');
});

process.on('SIGTERM', () => {
  void shutdown('SIGTERM');
});

app.get('/healthcheck', async () => {
  return { status: 'ok' };
});
