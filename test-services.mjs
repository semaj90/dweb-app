// Phase 4 Service Connection Test
import { createConnection } from 'net';

const services = [
    { name: 'Neo4j HTTP', port: 7474 },
    { name: 'Neo4j Bolt', port: 7687 },
    { name: 'PostgreSQL', port: 5432 },
    { name: 'Redis', port: 6379 },
    { name: 'RabbitMQ', port: 5672 },
    { name: 'Qdrant', port: 6333 },
    { name: 'Ollama', port: 11434 }
];

async function testService(service) {
    return new Promise((resolve) => {
        const socket = createConnection(service.port, 'localhost');

        socket.on('connect', () => {
            socket.destroy();
            resolve({ ...service, status: 'connected' });
        });

        socket.on('error', () => {
            resolve({ ...service, status: 'failed' });
        });

        setTimeout(() => {
            socket.destroy();
            resolve({ ...service, status: 'timeout' });
        }, 2000);
    });
}

console.log('🔍 Testing Phase 4 service connections...');

Promise.all(services.map(testService)).then(results => {
    results.forEach(result => {
        const status = result.status === 'connected' ? '✅' : '❌';
        console.log(`${status} ${result.name} (port ${result.port}): ${result.status}`);
    });

    const connectedCount = results.filter(r => r.status === 'connected').length;
    console.log(`\n📊 ${connectedCount}/${results.length} services connected`);

    process.exit(connectedCount > 0 ? 0 : 1);
});
