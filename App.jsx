import React, { useEffect, useRef, useState } from 'react';
import { createChart, AreaSeries } from 'lightweight-charts';
import { OptionsChain, PositionsTable } from "./OptionsChainModal";
import { SideTabs } from './SideTabs';

// --- Helper to parse date strings into Date objects for comparison ---
const parseSimDate = (dateVal) => {
  if (!dateVal) return new Date();
  if (typeof dateVal === 'number') {
    return new Date(dateVal > 1e11 ? dateVal : dateVal * 1000);
  }
  return new Date(dateVal);
};

// --- Helper to calculate Days To Expiration (DTE) relative to simulator date ---
const calculateSimDTE = (expirationStr, currentSimDateStr) => {
  if (!expirationStr || !currentSimDateStr) return 0;
  const exp = parseSimDate(expirationStr);
  const curr = parseSimDate(currentSimDateStr);
  
  // Set both to midnight to compare full calendar days
  exp.setHours(0, 0, 0, 0);
  curr.setHours(0, 0, 0, 0);

  const diffTime = exp.getTime() - curr.getTime();
  return Math.max(-1, Math.ceil(diffTime / (1000 * 60 * 60 * 24)));
};

// --- Main App Component ---
export default function App() {
  const chartContainerRef = useRef(null);
  const seriesRef = useRef(null);
  const chartRef = useRef(null);
  const lastTimeRef = useRef(0);
  
  const [currentPrice, setCurrentPrice] = useState(500.0);
  const [priceChange, setPriceChange] = useState(0);
  const [simulatedDate, setSimulatedDate] = useState(() => new Date().toISOString().slice(0, 10)); // Track simulator date
  const [logs, setLogs] = useState([]);
  
  // Track open and closed positions separately
  const [openPositions, setOpenPositions] = useState([]);
  const [closedPositions, setClosedPositions] = useState([]);
  const [activePositionsTab, setActivePositionsTab] = useState('OPEN');

  const [accountBalance, setAccountBalance] = useState(10000.0);

  const [npcStats, setNpcStats] = useState({
    bull: { portfolio_value: 10000, cash: 10000, contracts: 0 },
    bear: { portfolio_value: 10000, cash: 10000, contracts: 0 },
  });

  // Handle new order creation
  const handleExecuteTrade = (tradeData) => {
    const newPosition = {
      id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      entryDate: simulatedDate,
      timestamp: simulatedDate || new Date().toISOString(),
      ...tradeData, 
    };

    setOpenPositions((prev) => [newPosition, ...prev]);
    setAccountBalance((prev) => prev - tradeData.totalCost);
  };

  // Handle manual close from PositionsTable
  const handleClosePosition = (positionId, currentMark) => {
    const targetPos = openPositions.find((p) => p.id === positionId);
    if (!targetPos) return;

    const finalPayout = currentMark * 100 * targetPos.contracts;
    const realizedPnl = finalPayout - targetPos.totalCost;

    const closedItem = {
      ...targetPos,
      status: 'CLOSED',
      closeReason: 'MANUAL',
      exitPrice: currentMark,
      finalPayout,
      realizedPnl,
      closedAt: simulatedDate || new Date().toLocaleDateString(),
    };

    setOpenPositions((prev) => prev.filter((p) => p.id !== positionId));
    setClosedPositions((prev) => [closedItem, ...prev]);
    setAccountBalance((prev) => prev + finalPayout);
  };

  // Auto-close expired positions based on the advancing simulation date
  useEffect(() => {
    if (!openPositions || openPositions.length === 0 || !simulatedDate) return;

    const remainingOpen = [];
    const newlyClosed = [];
    let totalPayout = 0;

    openPositions.forEach((pos) => {
      // Calculate DTE dynamically relative to the current tick's simulation date
      const dte = calculateSimDTE(pos.expiration, simulatedDate);

      if (dte < 0) {
        let intrinsicPrice = 0;
        if (pos.type === 'CALL') {
          intrinsicPrice = Math.max(0, currentPrice - pos.strike);
        } else {
          intrinsicPrice = Math.max(0, pos.strike - currentPrice);
        }

        const finalPayout = intrinsicPrice * 100 * pos.contracts;
        const realizedPnl = finalPayout - pos.totalCost;

        totalPayout += finalPayout;
        newlyClosed.push({
          ...pos,
          status: 'CLOSED',
          closeReason: 'EXPIRED',
          exitPrice: intrinsicPrice,
          finalPayout,
          realizedPnl,
          closedAt: simulatedDate,
        });
      } else {
        // Update live DTE on the open position record
        remainingOpen.push({ ...pos, dte });
      }
    });

    if (newlyClosed.length > 0) {
      setOpenPositions(remainingOpen);
      setClosedPositions((prev) => [...newlyClosed, ...prev]);
      setAccountBalance((prev) => prev + totalPayout);
    }
  }, [simulatedDate, currentPrice]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      rightPriceScale: { borderColor: 'rgba(197, 203, 206, 0.8)' },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
        timeVisible: false,
        secondsVisible: false,
      },
      localization: { dateFormat: 'yyyy-MM-dd' }
    });

    const areaSeries = typeof chart.addAreaSeries === 'function'
      ? chart.addAreaSeries({
          topColor: 'rgba(38, 166, 154, 0.56)',
          bottomColor: 'rgba(38, 166, 154, 0.04)',
          lineColor: 'rgba(38, 166, 154, 1)',
          lineWidth: 2,
        })
      : chart.addSeries(AreaSeries, {
          topColor: 'rgba(38, 166, 154, 0.56)',
          bottomColor: 'rgba(38, 166, 154, 0.04)',
          lineColor: 'rgba(38, 166, 154, 1)',
          lineWidth: 2,
        });

    chartRef.current = chart;
    seriesRef.current = areaSeries;

    const ws = new WebSocket('ws://localhost:8000/ws/stocks');

    ws.onmessage = (event) => {
      try {
        const packet = JSON.parse(event.data);
        const price = packet.data?.SPY;
        const dateStr = packet.date || packet.timestamp || packet.data?.date;
        
        // if (dateStr) {
        //   setSimulatedDate(dateStr); // Synchronize simulation date across app
        // }
        
        if (dateStr) {
          // If dateStr is a full timestamp or number, normalize to YYYY-MM-DD
          const formattedDate = typeof dateStr === 'string' 
            ? dateStr.slice(0, 10) 
            : new Date(dateStr).toISOString().slice(0, 10);

          setSimulatedDate(formattedDate);
        }

        let timeInSeconds = typeof dateStr === 'number' 
          ? (dateStr > 1e11 ? Math.floor(dateStr / 1000) : dateStr)
          : Math.floor(new Date(dateStr).getTime() / 1000);

        if (price !== undefined && price !== null && !isNaN(timeInSeconds)) {
          if (timeInSeconds >= lastTimeRef.current) {
            lastTimeRef.current = timeInSeconds;
            areaSeries.update({ time: timeInSeconds, value: price });
          }

          setCurrentPrice((prev) => {
            if (prev !== null) setPriceChange(price - prev);
            return price;
          });
        }

        if (packet.npcs) {
          setNpcStats({
            bull: packet.npcs.bull || { portfolio_value: 0, cash: 0, contracts: 0 },
            bear: packet.npcs.bear || { portfolio_value: 0, cash: 0, contracts: 0 },
          });

          const bullAction = packet.npcs.bull?.action;
          const bearAction = packet.npcs.bear?.action;

          if (bullAction || bearAction) {
            setLogs((prevLogs) => {
              const newEntries = [];
              const lastLogDate = prevLogs.find((l) => l.date)?.date;
              if (packet.is_new_day || (lastLogDate && lastLogDate !== dateStr)) {
                newEntries.push({ id: `divider-${dateStr}-${Date.now()}`, isDivider: true, date: dateStr, text: ` ${dateStr}` });
              }
              if (bullAction) newEntries.push({ id: `${dateStr}-bull-${Math.random()}`, text: `Bull: ${bullAction}`, date: dateStr });
              if (bearAction) newEntries.push({ id: `${dateStr}-bear-${Math.random()}`, text: `Bear: ${bearAction}`, date: dateStr });

              return [...newEntries, ...prevLogs].slice(0, 25);
            });
          }
        }
      } catch (err) {
        console.error("Error parsing WebSocket packet:", err);
      }
    };

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current && chartContainerRef.current.clientWidth > 0) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // --- View: Dashboard Tab ---
  const dashboardView = (
    <>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid #1e293b', paddingBottom: '16px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            QuantWorld Sandbox
          </h1>
          <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: '14px' }}>GBM Simulator + RL Agent Speculation</p>
        </div>
        
        {currentPrice !== null && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#38bdf8', fontWeight: 'bold' }}>
              SIM DATE: {simulatedDate ? String(simulatedDate) : 'Connecting...'}
            </div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#f8fafc' }}>
              ${currentPrice.toFixed(2)}
              <span style={{ fontSize: '16px', marginLeft: '8px', color: priceChange >= 0 ? '#22c55e' : '#ef4444' }}>
                {priceChange >= 0 ? '▲' : '▼'} {Math.abs(priceChange).toFixed(2)}
              </span>
            </div>
          </div>
        )}
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '24px' }}>
        {/* Left Side: The Chart */}
        <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '16px', border: '1px solid #1e293b' }}>
          <h2 style={{ fontSize: '16px', margin: '0 0 12px 0', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
            Daily Market Price
          </h2>
          <div ref={chartContainerRef} style={{ width: '100%' }} />
        </div>

        {/* Right Side: Player Controls, NPC Metrics & Trading Log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #22c55e' }}>
              <div style={{ fontSize: '12px', color: '#22c55e', fontWeight: 'bold', textTransform: 'uppercase' }}>Bull NPC</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f8fafc', margin: '4px 0' }}>
                ${npcStats.bull.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Cash: ${npcStats.bull.cash?.toFixed(2)}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Pos: {npcStats.bull.contracts} Contract(s)</div>
            </div>

            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #ef4444' }}>
              <div style={{ fontSize: '12px', color: '#ef4444', fontWeight: 'bold', textTransform: 'uppercase' }}>Bear NPC</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f8fafc', margin: '4px 0' }}>
                ${npcStats.bear.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Cash: ${npcStats.bear.cash?.toFixed(2)}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Pos: {npcStats.bear.contracts} Contract(s)</div>
            </div>
          </div>

          <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '16px', border: '1px solid #1e293b', display: 'flex', flexDirection: 'column', height: '345px' }}>
            <h2 style={{ fontSize: '15px', margin: '0 0 12px 0', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              Agent & Player Activity
            </h2>
            <div style={{ flexGrow: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
              {logs.length === 0 ? (
                <p style={{ color: '#64748b', fontSize: '13px', fontStyle: 'italic' }}>Waiting for trade activity...</p>
              ) : (
                logs.map((log) => {
                  if (log.isDivider) {
                    return (
                      <div key={log.id} style={{ display: 'flex', alignItems: 'center', margin: '8px 0', opacity: 0.8 }}>
                        <div style={{ flexGrow: 1, borderTop: '1px solid #334155' }}></div>
                        <span style={{ padding: '2px 8px', fontSize: '10px', fontWeight: 'bold', color: '#38bdf8', backgroundColor: '#0f172a', borderRadius: '12px', border: '1px solid #1e293b' }}>
                          {log.text}
                        </span>
                        <div style={{ flexGrow: 1, borderTop: '1px solid #334155' }}></div>
                      </div>
                    );
                  }

                  const isPlayer = log.text.startsWith('Player:');
                  const isBuy = log.text.includes('BOUGHT');
                  
                  let borderClr = isBuy ? '#22c55e' : '#ef4444';
                  let bgClr = isBuy ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
                  if (isPlayer) { borderClr = '#38bdf8'; bgClr = 'rgba(56, 189, 248, 0.1)'; }

                  return (
                    <div key={log.id} style={{ padding: '8px', borderRadius: '4px', fontSize: '12px', backgroundColor: bgClr, borderLeft: `3px solid ${borderClr}`, color: '#e2e8f0' }}>
                      {log.text}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );

  // View: Closed Positions Table
  const closedPositionsView = (
    <div style={{ backgroundColor: '#131722', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' }}>
      {closedPositions.length === 0 ? (
        <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '13px' }}>No closed positions yet.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left', color: '#e2e8f0' }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #1e293b', backgroundColor: '#0f172a' }}>
                <th style={{ padding: '8px 12px' }}>CONTRACT</th>
                <th style={{ padding: '8px 12px' }}>QTY</th>
                <th style={{ padding: '8px 12px' }}>ENTRY COST</th>
                <th style={{ padding: '8px 12px' }}>EXIT PAYOUT</th>
                <th style={{ padding: '8px 12px' }}>REALIZED P&L</th>
                <th style={{ padding: '8px 12px' }}>CLOSED AT</th>
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
                    <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{pos.closedAt}</td>
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
        </div>
      )}
    </div>
  );

  return (
    <SideTabs
      currentPrice={currentPrice}
      accountBalance={accountBalance}
      positionsCount={openPositions.length}
      dashboardContent={dashboardView}
      optionsChainContent={
        <OptionsChain
          isOpen={true}
          currentPrice={currentPrice}
          simulatedDate={simulatedDate} // Pass simulation date to options chain
          onExecuteTrade={handleExecuteTrade}
        />
      }
      positionsContent={
        <div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            <button
              onClick={() => setActivePositionsTab('OPEN')}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                border: '1px solid #1e293b',
                backgroundColor: activePositionsTab === 'OPEN' ? '#1e293b' : '#131722',
                color: activePositionsTab === 'OPEN' ? '#38bdf8' : '#64748b',
                fontWeight: 'bold',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              Open Positions ({openPositions.length})
            </button>
            <button
              onClick={() => setActivePositionsTab('CLOSED')}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                border: '1px solid #1e293b',
                backgroundColor: activePositionsTab === 'CLOSED' ? '#1e293b' : '#131722',
                color: activePositionsTab === 'CLOSED' ? '#38bdf8' : '#64748b',
                fontWeight: 'bold',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              Closed History ({closedPositions.length})
            </button>
          </div>

          {activePositionsTab === 'OPEN' ? (
            <PositionsTable
              positions={openPositions}
              currentPrice={currentPrice}
              simulatedDate={simulatedDate} // Pass simulation date to table
              onClosePosition={handleClosePosition}
            />
          ) : (
            closedPositionsView
          )}
        </div>
      }
    />
  );
}