// new file: scripts/resolve-quic-ports.mjs
import net from 'net';

async function getFree(port, maxTries = 20) {
    for (let i = 0; i < maxTries; i++) {
        const p = port + i;
        const free = await new Promise(r => {
            const srv = net.createServer().once('error', () => r(false)).once('listening', () => { srv.close(() => r(true)); }).listen(p, '0.0.0.0');
        });
        if (free) return p;
    }
    throw new Error('No free port range found');
}

export async function resolveQuicPorts() {
    const LEGAL_QUIC = parseInt(process.env.LEGAL_QUIC_PORT || '8443', 10);
    const LEGAL_H3 = parseInt(process.env.LEGAL_H3_PORT || '8447', 10);
    const VECTOR_QUIC = parseInt(process.env.VECTOR_QUIC_PORT || '8543', 10);
    const VECTOR_H3 = parseInt(process.env.VECTOR_H3_PORT || '8545', 10);

    const resolved = {
        legalQuic: await getFree(LEGAL_QUIC),
        legalH3: await getFree(LEGAL_H3),
        vectorQuic: await getFree(VECTOR_QUIC),
        vectorH3: await getFree(VECTOR_H3)
    };
    Object.entries(resolved).forEach(([k, v]) => process.env[k.toUpperCase()] = String(v));
    console.log('✅ QUIC ports resolved:', resolved);
    return resolved;
}

if (import.meta.url === `file://${process.argv[1]}`) resolveQuicPorts();
