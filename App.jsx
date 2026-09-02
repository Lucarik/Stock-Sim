import React, { useEffect, useRef, useState } from 'react';
import { createChart, LineSeries, AreaSeries, CandlestickSeries } from 'lightweight-charts';
import { OptionsChain, PositionsTable } from "./OptionsChainModal";
import { SideTabs } from './SideTabs';

// Helper to parse date strings or unix timestamps into Date objects
const parseSimDate = (dateVal) => {
  if (!dateVal) return new Date();
  if (typeof dateVal === 'number') {
    return new Date(dateVal > 1e11 ? dateVal : dateVal * 1000);
  }
  return new Date(dateVal);
};

// Helper to calculate Days To Expiration (DTE) relative to simulator date
const calculateSimDTE = (expirationStr, currentSimDateStr) => {
  if (!expirationStr || !currentSimDateStr) return 0;
  const exp = parseSimDate(expirationStr);
  const curr = parseSimDate(currentSimDateStr);
  
  exp.setHours(0, 0, 0, 0);
  curr.setHours(0, 0, 0, 0);

  const diffTime = exp.getTime() - curr.getTime();
  return Math.max(-1, Math.ceil(diffTime / (1000 * 60 * 60 * 24)));
};

export default function App() {
  const chartContainerRef = useRef(null);
  const seriesRef = useRef(null);
  const chartRef = useRef(null);
  
  // Ticker Selection ('SPY' or 'QQQ')
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  
  // Ref to track current symbol inside persistent WebSocket callback
  const selectedSymbolRef = useRef(selectedSymbol);
  useEffect(() => {
    selectedSymbolRef.current = selectedSymbol;
  }, [selectedSymbol]);

  // Ref tracking to enforce strictly increasing timestamps for Lightweight Charts
  const spyLastTimeRef = useRef(0);
  const qqqLastTimeRef = useRef(0);

  // Persistent history refs
  const spyHistoryRef = useRef([]); // Stores SPY line data [{ time, value }]
  const qqqHistoryRef = useRef([]); // Stores QQQ candlestick data [{ time, open, high, low, close }]
  const currentCandleRef = useRef(null); // Tracks active QQQ candle

  // Multi-stock state tracking
  const [marketPrices, setMarketPrices] = useState({ SPY: 450.0, QQQ: 380.0 });
  const [priceChanges, setPriceChanges] = useState({ SPY: 0, QQQ: 0 });
  const [simulatedDates, setSimulatedDates] = useState({ SPY: '', QQQ: '' });
  
  const [logs, setLogs] = useState([]);
  
  // Player state
  const [openPositions, setOpenPositions] = useState([]);
  const [closedPositions, setClosedPositions] = useState([]);
  const [activePositionsTab, setActivePositionsTab] = useState('OPEN');
  const [accountBalance, setAccountBalance] = useState(10000.0);

  // Multi-stock NPC metrics
  const [npcStats, setNpcStats] = useState({
    SPY: {
      bull: { portfolio_value: 10000, cash: 10000, contracts: 0 },
      bear: { portfolio_value: 10000, cash: 10000, contracts: 0 },
    },
    QQQ: {
      bull: { portfolio_value: 10000, cash: 10000, contracts: 0 },
      bear: { portfolio_value: 10000, cash: 10000, contracts: 0 },
    }
  });

  const currentPrice = marketPrices[selectedSymbol] || 0;
  const priceChange = priceChanges[selectedSymbol] || 0;
  const simulatedDate = simulatedDates[selectedSymbol] || '';

  // Handle new trade execution
  const handleExecuteTrade = (tradeData) => {
    const newPosition = {
      id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      symbol: selectedSymbol,
      entryDate: simulatedDate,
      timestamp: simulatedDate || new Date().toISOString(),
      ...tradeData, 
    };

    setOpenPositions((prev) => [newPosition, ...prev]);
    setAccountBalance((prev) => prev - tradeData.totalCost);

    setLogs((prev) => [
      {
        id: `log_${Date.now()}`,
        text: `Player: BOUGHT ${tradeData.contracts}x ${selectedSymbol} $${tradeData.strike} ${tradeData.type}`,
        symbol: selectedSymbol,
      },
      ...prev,
    ]);
  };

  // Handle manual position close
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

  // Auto-close expired positions based on current simulation date
  useEffect(() => {
    if (!openPositions || openPositions.length === 0 || !simulatedDate) return;

    const remainingOpen = [];
    const newlyClosed = [];
    let totalPayout = 0;

    openPositions.forEach((pos) => {
      const activeDate = simulatedDates[pos.symbol] || simulatedDate;
      const dte = calculateSimDTE(pos.expiration, activeDate);
      const activePrice = marketPrices[pos.symbol] || currentPrice;

      if (dte < 0) {
        let intrinsicPrice = 0;
        if (pos.type === 'CALL') {
          intrinsicPrice = Math.max(0, activePrice - pos.strike);
        } else {
          intrinsicPrice = Math.max(0, pos.strike - activePrice);
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
          closedAt: activeDate,
        });
      } else {
        remainingOpen.push({ ...pos, dte });
      }
    });

    if (newlyClosed.length > 0) {
      setOpenPositions(remainingOpen);
      setClosedPositions((prev) => [...newlyClosed, ...prev]);
      setAccountBalance((prev) => prev + totalPayout);
    }
  }, [simulatedDates, marketPrices]);

  // --- 1. PERSISTENT WEBSOCKET CONNECTION (MOUNT ONCE) ---
  useEffect(() => {
    let isMounted = true;
    const ws = new WebSocket('ws://localhost:8000/ws/stocks');

    ws.onopen = () => {
      // If React unmounted while the handshake was in-flight, close cleanly now that it's OPEN
      if (!isMounted) {
        ws.close();
        return;
      }
      console.log('Connected to Stock WS');
    };

    ws.onmessage = (event) => {
      if (!isMounted) return;

      try {
        const packet = JSON.parse(event.data);

        if (packet.dates) {
          setSimulatedDates({
            SPY: packet.dates.SPY || '',
            QQQ: packet.dates.QQQ || '',
          });
        }

        if (packet.data) {
          setMarketPrices((prev) => {
            const nextSpy = packet.data.SPY ?? prev.SPY;
            const nextQqq = packet.data.QQQ ?? prev.QQQ;
            setPriceChanges({
              SPY: nextSpy - prev.SPY,
              QQQ: nextQqq - prev.QQQ,
            });
            return { SPY: nextSpy, QQQ: nextQqq };
          });
        }

        // --- UPDATE SPY HISTORY (Daily Line / Area Series) ---
        const spyPrice = packet.data?.SPY;
        const spyDateStr = packet.dates?.SPY; // Format: "YYYY-MM-DD"

        if (spyPrice !== undefined && spyPrice !== null && spyDateStr) {
          const history = spyHistoryRef.current;
          const lastPoint = history[history.length - 1];

          if (!lastPoint || lastPoint.time < spyDateStr) {
            // New date -> Push new point
            const newPoint = { time: spyDateStr, value: spyPrice };
            history.push(newPoint);

            if (selectedSymbolRef.current === 'SPY' && seriesRef.current) {
              seriesRef.current.update(newPoint);
            }
          } else if (lastPoint.time === spyDateStr) {
            // Same date -> Update today's active price
            const updatedPoint = { time: spyDateStr, value: spyPrice };
            history[history.length - 1] = updatedPoint;

            if (selectedSymbolRef.current === 'SPY' && seriesRef.current) {
              seriesRef.current.update(updatedPoint);
            }
          }
        }

        // --- UPDATE QQQ HISTORY (5-Minute Candlestick Series) ---
        const qqqPrice = packet.data?.QQQ;
        const qqqDateStr = packet.dates?.QQQ; // Format: "YYYY-MM-DD HH:mm:ss" or ISO

        if (qqqPrice !== undefined && qqqPrice !== null && qqqDateStr) {
          // Replace space with 'T' for reliable parsing across browsers
          const qqqTimeSec = Math.floor(new Date(qqqDateStr.replace(' ', 'T')).getTime() / 1000);

          if (!isNaN(qqqTimeSec) && qqqTimeSec > 0) {
            const bucketInterval = 300; // 5 minutes = 300s
            const candleTime = Math.floor(qqqTimeSec / bucketInterval) * bucketInterval;
            let activeCandle = currentCandleRef.current;

            // Reject out-of-order historical packets if they go backward in time
            if (!activeCandle || candleTime > activeCandle.time) {
              const prevClose = activeCandle ? activeCandle.close : qqqPrice;
              const newCandle = {
                time: candleTime,
                open: prevClose,
                high: Math.max(prevClose, qqqPrice),
                low: Math.min(prevClose, qqqPrice),
                close: qqqPrice,
              };
              
              qqqHistoryRef.current.push(newCandle);
              currentCandleRef.current = newCandle;

              if (selectedSymbolRef.current === 'QQQ' && seriesRef.current) {
                seriesRef.current.update(newCandle);
              }
            } else if (candleTime === activeCandle.time) {
              // Update ongoing active 5-minute candle
              const updatedCandle = {
                ...activeCandle,
                high: Math.max(activeCandle.high, qqqPrice),
                low: Math.min(activeCandle.low, qqqPrice),
                close: qqqPrice,
              };

              qqqHistoryRef.current[qqqHistoryRef.current.length - 1] = updatedCandle;
              currentCandleRef.current = updatedCandle;

              if (selectedSymbolRef.current === 'QQQ' && seriesRef.current) {
                seriesRef.current.update(updatedCandle);
              }
            }
          }
        }

        // --- NPC STATS ---
        if (packet.npcs) {
          setNpcStats({
            SPY: packet.npcs.SPY || { bull: {}, bear: {} },
            QQQ: packet.npcs.QQQ || { bull: {}, bear: {} },
          });
        }

        // --- TRADE ACTIVITY LOGS ---
        let incomingLogs = packet.logs || packet.trades || packet.npc_trades || packet.activity || packet.messages || [];

        if (packet.npcs) {
          ['SPY', 'QQQ'].forEach((sym) => {
            if (packet.npcs[sym]?.logs) incomingLogs = incomingLogs.concat(packet.npcs[sym].logs);
            if (packet.npcs[sym]?.trades) incomingLogs = incomingLogs.concat(packet.npcs[sym].trades);
            if (packet.npcs[sym]?.activity) incomingLogs = incomingLogs.concat(packet.npcs[sym].activity);
          });
        }

        if (incomingLogs && Array.isArray(incomingLogs) && incomingLogs.length > 0) {
          const formattedNewLogs = incomingLogs.map((item, idx) => {
            const rawText = typeof item === 'string' ? item : (item.text || item.message || item.trade || JSON.stringify(item));
            
            let logSymbol = item.symbol || item.ticker || item.asset;
            if (!logSymbol) {
              if (rawText.includes('QQQ')) logSymbol = 'QQQ';
              else if (rawText.includes('SPY')) logSymbol = 'SPY';
              else logSymbol = selectedSymbolRef.current;
            }

            return {
              id: `log_ws_${Date.now()}_${idx}_${Math.random().toString(36).substr(2, 4)}`,
              text: rawText,
              symbol: logSymbol,
              isDivider: item.isDivider || false
            };
          });

          setLogs((prev) => [...formattedNewLogs, ...prev].slice(0, 100));
        }

      } catch (err) {
        console.error("Error parsing WebSocket packet:", err);
      }
    };

    return () => {
      isMounted = false;
      ws.onmessage = null;

      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else if (ws.readyState === WebSocket.CONNECTING) {
        // Defer closing until the connection finishes establishing to prevent browser console warning
        ws.onopen = () => ws.close();
      }
    };
  }, []);

  // --- 2. RENDER CHART UPON SYMBOL SELECTION ---
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
        timeVisible: true,
        secondsVisible: true,
      },
      localization: { dateFormat: 'yyyy-MM-dd' }
    });

    let series;
    if (selectedSymbol === 'SPY') {
      series = typeof chart.addAreaSeries === 'function'
        ? chart.addAreaSeries({
            topColor: 'rgba(38, 166, 154, 0.56)',
            bottomColor: 'rgba(38, 166, 154, 0.04)',
            lineColor: '#26a69a',
            lineWidth: 2,
          })
        : chart.addSeries(LineSeries, { color: '#26a69a', lineWidth: 2 });

      if (spyHistoryRef.current.length > 0) {
        series.setData(spyHistoryRef.current);
      }
    } else {
      series = typeof chart.addCandlestickSeries === 'function'
        ? chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
          })
        : chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
          });

      if (qqqHistoryRef.current.length > 0) {
        series.setData(qqqHistoryRef.current);
      }
    }

    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current && chartContainerRef.current.clientWidth > 0) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      seriesRef.current = null;
      chartRef.current = null;
      chart.remove();
    };
  }, [selectedSymbol]);

  const activeNpc = npcStats[selectedSymbol] || { bull: {}, bear: {} };

  // Filter logs for selected ticker
  const filteredLogs = logs.filter(
    (log) => !log.symbol || log.symbol === selectedSymbol
  );

  // --- View: Dashboard Tab ---
  const dashboardView = (
    <>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid #1e293b', paddingBottom: '16px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            QuantWorld Sandbox
          </h1>
          <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: '14px' }}>Multi-Asset GBM Simulator + RL Agent Speculation</p>
        </div>

        {/* Ticker Selector Buttons */}
        <div style={{ display: 'flex', gap: '8px', backgroundColor: '#0f172a', padding: '4px', borderRadius: '8px', border: '1px solid #1e293b' }}>
          {['SPY', 'QQQ'].map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              style={{
                padding: '6px 16px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: selectedSymbol === symbol ? '#38bdf8' : 'transparent',
                color: selectedSymbol === symbol ? '#0f172a' : '#94a3b8',
                fontWeight: 'bold',
                cursor: 'pointer',
                fontSize: '13px',
                transition: 'all 0.2s',
              }}
            >
              {symbol} {symbol === 'QQQ' ? '(5m Candles)' : '(Ticks)'}
            </button>
          ))}
        </div>
        
        {currentPrice !== null && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#38bdf8', fontWeight: 'bold' }}>
              SIM DATE ({selectedSymbol}): {simulatedDate ? String(simulatedDate) : 'Connecting...'}
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
        {/* Left Side: Chart */}
        <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '16px', border: '1px solid #1e293b' }}>
          <h2 style={{ fontSize: '16px', margin: '0 0 12px 0', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
            {selectedSymbol} {selectedSymbol === 'QQQ' ? 'Candlestick Chart (5-min)' : 'Area Chart'}
          </h2>
          <div ref={chartContainerRef} style={{ width: '100%' }} />
        </div>

        {/* Right Side: Agent Metrics & Activity Log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #22c55e' }}>
              <div style={{ fontSize: '12px', color: '#22c55e', fontWeight: 'bold', textTransform: 'uppercase' }}>Bull Agent ({selectedSymbol})</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f8fafc', margin: '4px 0' }}>
                ${activeNpc.bull?.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '10,000.00'}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Cash: ${activeNpc.bull?.cash?.toFixed(2) || '10000.00'}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Pos: {activeNpc.bull?.contracts || 0} Contract(s)</div>
            </div>

            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #ef4444' }}>
              <div style={{ fontSize: '12px', color: '#ef4444', fontWeight: 'bold', textTransform: 'uppercase' }}>Bear Agent ({selectedSymbol})</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f8fafc', margin: '4px 0' }}>
                ${activeNpc.bear?.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '10,000.00'}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Cash: ${activeNpc.bear?.cash?.toFixed(2) || '10000.00'}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Pos: {activeNpc.bear?.contracts || 0} Contract(s)</div>
            </div>
          </div>

          <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '16px', border: '1px solid #1e293b', display: 'flex', flexDirection: 'column', height: '345px' }}>
            <h2 style={{ fontSize: '15px', margin: '0 0 12px 0', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              {selectedSymbol} Activity Log
            </h2>
            <div style={{ flexGrow: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
              {filteredLogs.length === 0 ? (
                <p style={{ color: '#64748b', fontSize: '13px', fontStyle: 'italic' }}>Waiting for trade activity...</p>
              ) : (
                filteredLogs.map((log) => {
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
                  const isBuy = log.text.includes('BOUGHT') || log.text.includes('BUY');
                  
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
                <th style={{ padding: '8px 12px' }}>TICKER</th>
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
                    <td style={{ padding: '10px 12px', fontWeight: 'bold', color: '#38bdf8' }}>{pos.symbol || 'SPY'}</td>
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
          symbol={selectedSymbol}
          currentPrice={currentPrice}
          simulatedDate={simulatedDate}
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
              simulatedDate={simulatedDate}
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