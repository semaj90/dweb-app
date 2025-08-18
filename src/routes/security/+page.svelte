<script lang="ts">
  import { onMount } from 'svelte';
  
  // Security monitoring state
  let securityStatus = $state({
    threats: 0,
    blockedIPs: 47,
    activeSessions: 3,
    lastScan: 'OPERATIONAL',
    systemIntegrity: 'SECURE',
    accessControl: 'ENABLED',
    encryption: 'AES-256 ACTIVE'
  });

  let recentActivity = $state([
    { time: '2024-01-15 14:23:45', event: 'User login', user: 'admin', status: 'SUCCESS', ip: '192.168.1.100' },
    { time: '2024-01-15 14:22:12', event: 'System scan', user: 'SYSTEM', status: 'COMPLETE', ip: 'localhost' },
    { time: '2024-01-15 14:20:33', event: 'API access', user: 'ai-service', status: 'SUCCESS', ip: '127.0.0.1' },
    { time: '2024-01-15 14:18:44', event: 'Database backup', user: 'SYSTEM', status: 'SUCCESS', ip: 'localhost' }
  ]);

  let threatLevel = $derived(() => {
    if (securityStatus.threats === 0) return 'LOW';
    if (securityStatus.threats < 5) return 'MEDIUM';
    return 'HIGH';
  });

  let isScanning = $state(false);

  function performSecurityScan() {
    isScanning = true;
    setTimeout(() => {
      isScanning = false;
      securityStatus.lastScan = new Date().toLocaleTimeString();
      // Simulate scan results
      const newActivity = {
        time: new Date().toLocaleString(),
        event: 'Security scan',
        user: 'SYSTEM',
        status: 'COMPLETE',
        ip: 'localhost'
      };
      recentActivity = [newActivity, ...recentActivity].slice(0, 10);
    }, 3000);
  }

  onMount(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        securityStatus.activeSessions = Math.floor(Math.random() * 10) + 1;
      }
    }, 5000);

    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>YoRHa Security Center</title>
</svelte:head>

<div class="security-center">
  <div class="security-header">
    <h1 class="page-title">
      <span class="title-icon">🛡️</span>
      YORHA SECURITY CENTER
    </h1>
    <div class="threat-level">
      <span class="threat-label">THREAT LEVEL:</span>
      <span class="threat-value {threatLevel.toLowerCase()}">{threatLevel}</span>
    </div>
  </div>

  <div class="security-grid">
    <!-- System Status Overview -->
    <section class="status-overview">
      <h2 class="section-title">SYSTEM STATUS</h2>
      <div class="status-grid">
        <div class="status-item">
          <div class="status-icon">🚨</div>
          <div class="status-content">
            <div class="status-label">THREATS DETECTED</div>
            <div class="status-value">{securityStatus.threats}</div>
          </div>
        </div>
        <div class="status-item">
          <div class="status-icon">🚫</div>
          <div class="status-content">
            <div class="status-label">BLOCKED IPS</div>
            <div class="status-value">{securityStatus.blockedIPs}</div>
          </div>
        </div>
        <div class="status-item">
          <div class="status-icon">👥</div>
          <div class="status-content">
            <div class="status-label">ACTIVE SESSIONS</div>
            <div class="status-value">{securityStatus.activeSessions}</div>
          </div>
        </div>
        <div class="status-item">
          <div class="status-icon">🔐</div>
          <div class="status-content">
            <div class="status-label">ENCRYPTION</div>
            <div class="status-value">ACTIVE</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Security Controls -->
    <section class="security-controls">
      <h2 class="section-title">SECURITY CONTROLS</h2>
      <div class="controls-grid">
        <button 
          class="control-button scan-button {isScanning ? 'scanning' : ''}"
          onclick={performSecurityScan}
          disabled={isScanning}
        >
          <div class="button-icon">🔍</div>
          <div class="button-text">
            {isScanning ? 'SCANNING...' : 'RUN SECURITY SCAN'}
          </div>
        </button>
        
        <button class="control-button">
          <div class="button-icon">🔄</div>
          <div class="button-text">UPDATE FIREWALL</div>
        </button>
        
        <button class="control-button">
          <div class="button-icon">📊</div>
          <div class="button-text">GENERATE REPORT</div>
        </button>
        
        <button class="control-button">
          <div class="button-icon">⚙️</div>
          <div class="button-text">CONFIGURE RULES</div>
        </button>
      </div>
    </section>

    <!-- Recent Activity -->
    <section class="recent-activity">
      <h2 class="section-title">RECENT SECURITY ACTIVITY</h2>
      <div class="activity-list">
        {#each recentActivity as activity}
          <div class="activity-item">
            <div class="activity-time">{activity.time}</div>
            <div class="activity-event">{activity.event}</div>
            <div class="activity-user">{activity.user}</div>
            <div class="activity-status {activity.status.toLowerCase()}">{activity.status}</div>
            <div class="activity-ip">{activity.ip}</div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Security Metrics -->
    <section class="security-metrics">
      <h2 class="section-title">SECURITY METRICS</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-title">SYSTEM INTEGRITY</div>
          <div class="metric-value secure">{securityStatus.systemIntegrity}</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">ACCESS CONTROL</div>
          <div class="metric-value active">{securityStatus.accessControl}</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">LAST SCAN</div>
          <div class="metric-value">{securityStatus.lastScan}</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">ENCRYPTION</div>
          <div class="metric-value active">{securityStatus.encryption}</div>
        </div>
      </div>
    </section>
  </div>

  <!-- Back to Command Center -->
  <div class="navigation-footer">
    <a href="/" class="back-button">
      <span class="button-icon">⬅️</span>
      RETURN TO COMMAND CENTER
    </a>
  </div>
</div>

<style>
  .security-center {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .security-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #ffbf00;
    border-radius: 8px;
  }

  .page-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    background: linear-gradient(45deg, #ffbf00, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #ffbf00);
  }

  .threat-level {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
  }

  .threat-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  .threat-value {
    font-size: 1.5rem;
    font-weight: 700;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    letter-spacing: 2px;
  }

  .threat-value.low {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    border: 1px solid #00ff41;
  }

  .threat-value.medium {
    background: rgba(255, 191, 0, 0.2);
    color: #ffbf00;
    border: 1px solid #ffbf00;
  }

  .threat-value.high {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .security-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffbf00;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #ffbf00;
  }

  /* Status Overview */
  .status-overview {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
  }

  .status-item:hover {
    background: rgba(255, 191, 0, 0.1);
    border-color: #ffbf00;
  }

  .status-icon {
    font-size: 2rem;
  }

  .status-content {
    flex: 1;
  }

  .status-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-bottom: 0.5rem;
  }

  .status-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00ff41;
  }

  /* Security Controls */
  .security-controls {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .controls-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .control-button {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #ffbf00, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1.5rem;
    border-radius: 6px;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .control-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(255, 191, 0, 0.3);
  }

  .control-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .control-button.scanning {
    animation: pulse 1.5s ease-in-out infinite;
  }

  .button-icon {
    font-size: 1.5rem;
  }

  .button-text {
    font-size: 0.9rem;
    text-align: center;
  }

  /* Recent Activity */
  .recent-activity {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .activity-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .activity-item {
    display: grid;
    grid-template-columns: 200px 1fr 150px 100px 150px;
    gap: 1rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    align-items: center;
  }

  .activity-time {
    color: #4ecdc4;
    font-weight: 600;
  }

  .activity-event {
    color: #f0f0f0;
  }

  .activity-user {
    color: #ffbf00;
    font-weight: 600;
  }

  .activity-status {
    font-weight: 700;
    text-align: center;
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.8rem;
  }

  .activity-status.success {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    border: 1px solid #00ff41;
  }

  .activity-status.complete {
    background: rgba(78, 205, 196, 0.2);
    color: #4ecdc4;
    border: 1px solid #4ecdc4;
  }

  .activity-ip {
    color: #f0f0f0;
    opacity: 0.7;
  }

  /* Security Metrics */
  .security-metrics {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .metric-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
  }

  .metric-title {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .metric-value {
    font-size: 1.3rem;
    font-weight: 700;
  }

  .metric-value.secure {
    color: #00ff41;
  }

  .metric-value.active {
    color: #4ecdc4;
  }

  /* Navigation Footer */
  .navigation-footer {
    text-align: center;
    margin-top: 3rem;
  }

  .back-button {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(145deg, #ffbf00, #00ff41);
    color: #0a0a0a;
    padding: 1rem 2rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(255, 191, 0, 0.3);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .security-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .security-grid {
      grid-template-columns: 1fr;
    }

    .activity-item {
      grid-template-columns: 1fr;
      gap: 0.5rem;
      text-align: center;
    }

    .controls-grid {
      grid-template-columns: 1fr;
    }
  }
</style>