<!-- NieR-themed navigation component -->
<nav class="nier-nav">
  <div class="nier-nav-container">
    <div class="nier-brand">
      <span class="nier-brand-text">LEGAL AI SYSTEM</span>
      <span class="nier-brand-version">v3.0.0</span>
    </div>
    
    <div class="nier-nav-links">
      <a href="/cases" class="nier-nav-link">CASES</a>
      <a href="/evidence" class="nier-nav-link">EVIDENCE</a>
      <a href="/reports" class="nier-nav-link">REPORTS</a>
      <a href="/analytics" class="nier-nav-link">ANALYTICS</a>
    </div>
    
    <div class="nier-nav-actions">
      <button class="nier-toolbar-btn">⚙</button>
      <button class="nier-toolbar-btn">⚡</button>
    </div>
  </div>
</nav>

<style>
  /* @unocss-include */
.nier-nav {
  background: var(--nier-bg-secondary);
  border-bottom: var(--nier-border-width) solid var(--nier-border-primary);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nier-nav-container {
  align-items: center;
  display: flex;
  justify-content: space-between;
  max-width: 1200px;
  margin: 0 auto;
  padding: var(--nier-space-md) var(--nier-space-lg);
}

.nier-brand {
  display: flex;
  flex-direction: column;
  gap: var(--nier-space-xs);
}

.nier-brand-text {
  color: var(--nier-accent-primary);
  font-family: var(--nier-font-family);
  font-size: var(--nier-font-size-lg);
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.nier-brand-version {
  color: var(--nier-text-muted);
  font-size: var(--nier-font-size-sm);
  letter-spacing: 1px;
}

.nier-nav-links {
  display: flex;
  gap: var(--nier-space-lg);
}

.nier-nav-link {
  color: var(--nier-text-secondary);
  font-family: var(--nier-font-family);
  font-size: var(--nier-font-size-sm);
  letter-spacing: 1px;
  text-decoration: none;
  text-transform: uppercase;
  transition: color 0.2s ease;
}

.nier-nav-link:hover {
  color: var(--nier-accent-primary);
}

.nier-nav-actions {
  display: flex;
  gap: var(--nier-space-sm);
}
</style>
