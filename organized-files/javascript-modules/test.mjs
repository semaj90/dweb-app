import { createConnection } from 'net';

const services = [
    { name: 'PostgreSQL', port: 5432, required: true },
    { name: 'Redis', port: 6379, required: true },
    { name: 'Neo4j', port: 7474, required: true },
    { name: 'Qdrant', port: 6333, required: true },
    { name: 'RabbitMQ', port: 15672, required: true },
    { name: 'TTS', port: 5002, required: true }
];

async function testService(service) {
    return new Promise((resolve) => {
        const socket = createConnection(service.port, 'localhost');
        const timeout = setTimeout(() => {
            socket.destroy();
            resolve({ ...service, status: 'timeout' });
        }, 3000);
        
        socket.on('connect', () => {
            clearTimeout(timeout);
            socket.destroy();
            resolve({ ...service, status: 'connected' });
        });
        
        socket.on('error', () => {
            clearTimeout(timeout);
            resolve({ ...service, status: 'failed' });
        });
    });
}

console.log('🧪 Testing Legal AI System Integration...');
console.log('=' + '='.repeat(50));

const results = await Promise.all(services.map(testService));

console.log('\n📊 Service Connectivity Results:');
results.forEach(result => {
    const icon = result.status === 'connected' ? '✅' : 
                result.required ? '❌' : '⚠️';
    console.log(`${icon} ${result.name}: ${result.status}`);
});

const connected = results.filter(r => r.status === 'connected').length;
const required = results.filter(r => r.required).length;
const connectedRequired = results.filter(r => r.required && r.status === 'connected').length;

console.log(`\n📈 Summary: ${connected}/${services.length} total, ${connectedRequired}/${required} required`);

if (connectedRequired === required) {
    console.log('\n🎉 INTEGRATION TEST: SUCCESS!');
    console.log('✅ All required services operational');
    console.log('🚀 Ready for Phase 5 development');
    console.log('\n🌐 Access URLs:');
    console.log('• Neo4j Browser: http://localhost:7474 (neo4j/LegalRAG2024!)');
    console.log('• RabbitMQ Management: http://localhost:15672 (legal_admin/LegalRAG2024!)');
    console.log('• Qdrant Dashboard: http://localhost:6333');
    console.log('• TTS Service: http://localhost:5002/health');
    process.exit(0);
} else {
    console.log('\n⚠️ INTEGRATION TEST: Issues detected');
    console.log('❌ Some required services not accessible');
    console.log('\n💡 Try: docker-compose up -d && wait 30 seconds');
    process.exit(1);
}
