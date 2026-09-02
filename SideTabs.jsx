import React, { useState } from 'react';

export function SideTabs({ 
  positionsCount = 0, 
  dashboardContent,
  positionsContent, 
  optionsChainContent,
  accountBalance,
  currentPrice
}) {
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'chain' | 'positions'

  return (
    <div style={styles.container}>
      {/* --- Sidebar Navigation --- */}
      <aside style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <span style={styles.brandTitle}>QUANTWORLD</span>
          <div style={styles.tickerBadge}>
            <span style={styles.dot} /> LIVE
          </div>
        </div>

        {/* Account Info Pill */}
        <div style={styles.accountCard}>
          <div style={styles.accountItem}>
            <span style={styles.accountLabel}>SPY SPOT</span>
            <span style={styles.spotVal}>
              {currentPrice ? `$${Number(currentPrice).toFixed(2)}` : '---'}
            </span>
          </div>
          <div style={styles.accountItem}>
            <span style={styles.accountLabel}>BALANCE</span>
            <span style={styles.balanceVal}>
              ${accountBalance ? Number(accountBalance).toLocaleString(undefined, { minimumFractionDigits: 2 }) : '10,000.00'}
            </span>
          </div>
        </div>

        {/* Tab Buttons */}
        <nav style={styles.navGroup}>
          <button
            onClick={() => setActiveTab('dashboard')}
            style={{ ...styles.tabBtn, ...(activeTab === 'dashboard' ? styles.tabBtnActive : {}) }}
          >
            <span style={styles.tabIcon}>📈</span>
            <span>Dashboard</span>
          </button>

          <button
            onClick={() => setActiveTab('chain')}
            style={{ ...styles.tabBtn, ...(activeTab === 'chain' ? styles.tabBtnActive : {}) }}
          >
            <span style={styles.tabIcon}>⚡</span>
            <span>Options Chain</span>
          </button>

          <button
            onClick={() => setActiveTab('positions')}
            style={{ ...styles.tabBtn, ...(activeTab === 'positions' ? styles.tabBtnActive : {}) }}
          >
            <span style={styles.tabIcon}>📊</span>
            <span>Positions</span>
            {positionsCount > 0 && (
              <span style={styles.badge}>{positionsCount}</span>
            )}
          </button>
        </nav>
      </aside>

      {/* --- Main Content Area --- */}
      {/* We use display: block/none instead of conditional rendering so the lightweight-chart canvas isn't destroyed when switching tabs! */}
      <main style={styles.mainContent}>
        <div style={{ ...styles.tabPanel, display: activeTab === 'dashboard' ? 'block' : 'none' }}>
          {dashboardContent}
        </div>

        <div style={{ ...styles.tabPanel, display: activeTab === 'chain' ? 'block' : 'none' }}>
          <div style={styles.panelHeader}>
            <h3 style={styles.panelTitle}>Options Chain</h3>
            <p style={styles.panelSub}>Execute dynamic orders across strike prices</p>
          </div>
          {optionsChainContent}
        </div>

        <div style={{ ...styles.tabPanel, display: activeTab === 'positions' ? 'block' : 'none' }}>
          <div style={styles.panelHeader}>
            <h3 style={styles.panelTitle}>Active Positions & Orders</h3>
            <p style={styles.panelSub}>Monitor live P&L and close open contracts</p>
          </div>
          {positionsContent}
        </div>
      </main>
    </div>
  );
}

// --- Inline Styles for Dark Theme Sidebar ---
const styles = {
  container: {
    display: 'flex',
    minHeight: '100vh',
    backgroundColor: '#0b0e14',
    color: '#e2e8f0',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  sidebar: {
    width: '170px',
    backgroundColor: '#131722',
    borderRight: '1px solid #1e293b',
    padding: '20px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    flexShrink: 0
  },
  sidebarHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: '12px',
    borderBottom: '1px solid #1e293b'
  },
  brandTitle: {
    fontSize: '13px',
    fontWeight: '800',
    letterSpacing: '0.8px',
    color: '#38bdf8'
  },
  tickerBadge: {
    fontSize: '10px',
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    color: '#22c55e',
    padding: '2px 6px',
    borderRadius: '4px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '4px'
  },
  dot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: '#22c55e'
  },
  accountCard: {
    backgroundColor: '#0f172a',
    borderRadius: '6px',
    padding: '10px 12px',
    border: '1px solid #1e293b',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  accountItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  accountLabel: {
    fontSize: '10px',
    color: '#64748b',
    fontWeight: '700'
  },
  spotVal: {
    fontSize: '13px',
    fontWeight: 'bold',
    color: '#38bdf8'
  },
  balanceVal: {
    fontSize: '13px',
    fontWeight: 'bold',
    color: '#22c55e'
  },
  navGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px'
  },
  tabBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    width: '100%',
    padding: '10px 12px',
    backgroundColor: 'transparent',
    border: 'none',
    borderRadius: '6px',
    color: '#94a3b8',
    fontSize: '13px',
    fontWeight: '600',
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'all 0.15s ease'
  },
  tabBtnActive: {
    backgroundColor: '#1e293b',
    color: '#f8fafc',
    borderLeft: '3px solid #38bdf8'
  },
  tabIcon: {
    fontSize: '14px'
  },
  badge: {
    marginLeft: 'auto',
    backgroundColor: '#38bdf8',
    color: '#0f172a',
    fontSize: '10px',
    fontWeight: '800',
    borderRadius: '10px',
    padding: '2px 7px'
  },
  mainContent: {
    flex: 1,
    padding: '24px 32px',
    overflowY: 'auto'
  },
  tabPanel: {
    maxWidth: '1400px',
    margin: '0 auto'
  },
  panelHeader: {
    marginBottom: '20px'
  },
  panelTitle: {
    margin: 0,
    fontSize: '20px',
    fontWeight: '700',
    color: '#f8fafc'
  },
  panelSub: {
    margin: '4px 0 0 0',
    fontSize: '12px',
    color: '#64748b'
  }
};