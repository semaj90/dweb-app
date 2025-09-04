// run-cluster-manager.mjs
// Cluster manager orchestration using central port allocator
import { ClusterMulticoreManager } from './cluster-multicore-manager.mjs';

const manager = new ClusterMulticoreManager();

const portRanges = {
  clusterManager: { start: 3001, end: 3010 },
  loadBalancer:   { start: 8099, end: 8099 }
};

(async () => {
  const ports = await manager.reservePorts(portRanges);
  console.log('Cluster ports assigned:', ports);

  // TODO: Add cluster manager and load balancer spawn logic here
  // Example:
  // spawnClusterManager(ports.clusterManager);
  // spawnLoadBalancer(ports.loadBalancer);
})();
