import React, { useState, useEffect } from 'react';

export function PositionsManager({ currentPrice, openPositions, setOpenPositions, closedPositions, setClosedPositions }) {
  const [activeTab, setActiveTab] = useState('OPEN'); // 'OPEN' or 'CLOSED'

  // Helper: Calculate intrinsic value at expiration
  const calculateExpirationPayout = (pos, spotPrice) => {
    if (pos.type === 'CALL') {
      return Math.max(0, spotPrice - pos.strike) * 100 * pos.contracts;
    } else {
      return Math.max(0, pos.strike - spotPrice) * 100 * pos.contracts;
    }
  };

  // Auto-close expired positions when currentPrice or openPositions update
  useEffect(() => {
    if (!openPositions || openPositions.length === 0) return;

    const remainingOpen = [];
    const newlyClosed = [];

    openPositions.forEach((pos) => {
      // Condition: DTE reaches 0 or expiration date is reached
      const isExpired = pos.dte <= 0;

      if (isExpired) {
        const finalPayout = calculateExpirationPayout(pos, currentPrice);
        const realizedPnl = finalPayout - pos.totalCost;

        newlyClosed.push({
          ...pos,
          status: 'CLOSED',
          closeReason: 'EXPIRED',
          exitPrice: finalPayout / (100 * pos.contracts),
          finalPayout,
          realizedPnl,
          closedAt: new Date().toLocaleDateString(),
        });
      } else {
        remainingOpen.push(pos);
      }
    });

    // If any positions expired, update both state arrays
    if (newlyClosed.length > 0) {
      setOpenPositions(remainingOpen);
      setClosedPositions((prev) => [...newlyClosed, ...prev]);
    }
  }, [currentPrice, openPositions, setOpenPositions, setClosedPositions]);

  // Manual Close Handler
  const handleManualClose = (posToClose) => {
    const currentContractValue = (posToClose.bid || posToClose.premium) * 100 * posToClose.contracts;
    const realizedPnl = currentContractValue - posToClose.totalCost;

    const closedItem = {
      ...posToClose,
      status: 'CLOSED',
      closeReason: 'MANUAL',
      exitPrice: posToClose.bid || posToClose.premium,
      finalPayout: currentContractValue,
      realizedPnl,
      closedAt: new Date().toLocaleDateString(),
    };

    setOpenPositions((prev) => prev.filter((p) => p.id !== posToClose.id));
    setClosedPositions((prev) => [closedItem, ...prev]);
  };

  return (
    <div style={{ backgroundColor: '#131722', border: '1px solid #1e293b', borderRadius: '8px', padding: '20px', color: '#e2e8f0' }}>
      {/* Header & Tabs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid #1e293b', paddingBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold' }}>Positions</h3>

        <div style={{ display: 'flex', gap: '8px', backgroundColor: '#0b0e14', padding: '4px', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <button
            onClick={() => setActiveTab('OPEN')}
            style={{
              padding: '6px 14px',
              borderRadius: '4px',
              border: 'none',
              backgroundColor: activeTab === 'OPEN' ? '#1e293b' : 'transparent',
              color: activeTab === 'OPEN' ? '#38bdf8' : '#64748b',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '12px',
            }}
          >
            Open ({openPositions.length})
          </button>
          <button
            onClick={() => setActiveTab('CLOSED')}
            style={{
              padding: '6px 14px',
              borderRadius: '4px',
              border: 'none',
              backgroundColor: activeTab === 'CLOSED' ? '#1e293b' : 'transparent',
              color: activeTab === 'CLOSED' ? '#38bdf8' : '#64748b',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '12px',
            }}
          >
            Closed ({closedPositions.length})
          </button>
        </div>
      </div>

      {/* OPEN POSITIONS TABLE */}
      {activeTab === 'OPEN' && (
        <div style={{ overflowX: 'auto' }}>
          {openPositions.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>No active open positions.</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #1e293b', backgroundColor: '#0f172a' }}>
                  <th style={{ padding: '8px 12px' }}>CONTRACT</th>
                  <th style={{ padding: '8px 12px' }}>QTY</th>
                  <th style={{ padding: '8px 12px' }}>AVG PRICE</th>
                  <th style={{ padding: '8px 12px' }}>TOTAL COST</th>
                  <th style={{ padding: '8px 12px' }}>DTE</th>
                  <th style={{ padding: '8px 12px' }}>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {openPositions.map((pos) => (
                  <tr key={pos.id || `${pos.strike}-${pos.type}-${pos.expiration}`} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 'bold' }}>
                      <span style={{ color: pos.type === 'CALL' ? '#22c55e' : '#ef4444', marginRight: '6px' }}>{pos.type}</span>
                      ${pos.strike} ({pos.expiration})
                    </td>
                    <td style={{ padding: '10px 12px' }}>{pos.contracts}</td>
                    <td style={{ padding: '10px 12px' }}>${Number(pos.premium).toFixed(2)}</td>
                    <td style={{ padding: '10px 12px' }}>${Number(pos.totalCost).toFixed(2)}</td>
                    <td style={{ padding: '10px 12px', color: pos.dte <= 1 ? '#ef4444' : '#38bdf8', fontWeight: 'bold' }}>
                      {pos.dte}d
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <button
                        onClick={() => handleManualClose(pos)}
                        style={{
                          padding: '4px 10px',
                          backgroundColor: '#1e293b',
                          color: '#f8fafc',
                          border: '1px solid #334155',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '11px',
                        }}
                      >
                        Close Early
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* CLOSED POSITIONS TABLE */}
      {activeTab === 'CLOSED' && (
        <div style={{ overflowX: 'auto' }}>
          {closedPositions.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>No closed history available.</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #1e293b', backgroundColor: '#0f172a' }}>
                  <th style={{ padding: '8px 12px' }}>CONTRACT</th>
                  <th style={{ padding: '8px 12px' }}>QTY</th>
                  <th style={{ padding: '8px 12px' }}>ENTRY COST</th>
                  <th style={{ padding: '8px 12px' }}>EXIT PAYOUT</th>
                  <th style={{ padding: '8px 12px' }}>REALIZED P&L</th>
                  <th style={{ padding: '8px 12px' }}>REASON</th>
                </tr>
              </thead>
              <tbody>
                {closedPositions.map((pos, idx) => {
                  const isProfit = pos.realizedPnl >= 0;
                  return (
                    <tr key={pos.id || idx} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 'bold' }}>
                        <span style={{ color: pos.type === 'CALL' ? '#22c55e' : '#ef4444', marginRight: '6px' }}>{pos.type}</span>
                        ${pos.strike} ({pos.expiration})
                      </td>
                      <td style={{ padding: '10px 12px' }}>{pos.contracts}</td>
                      <td style={{ padding: '10px 12px' }}>${Number(pos.totalCost).toFixed(2)}</td>
                      <td style={{ padding: '10px 12px' }}>${Number(pos.finalPayout).toFixed(2)}</td>
                      <td style={{ padding: '10px 12px', fontWeight: 'bold', color: isProfit ? '#22c55e' : '#ef4444' }}>
                        {isProfit ? '+' : ''}${Number(pos.realizedPnl).toFixed(2)}
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <span
                          style={{
                            padding: '2px 6px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: 'bold',
                            backgroundColor: pos.closeReason === 'EXPIRED' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                            color: pos.closeReason === 'EXPIRED' ? '#ef4444' : '#38bdf8',
                          }}
                        >
                          {pos.closeReason}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}