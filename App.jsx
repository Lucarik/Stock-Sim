import React, { useEffect, useRef, useState } from 'react';
import { createChart, LineSeries, CandlestickSeries } from 'lightweight-charts';
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

// Check if timestamp string represents intraday (has non-zero time)
const isIntradayDate = (dateStr) => {
  if (!dateStr || typeof dateStr !== 'string') return false;
  if (!dateStr.includes(':')) return false;
  const timePart = dateStr.split(' ')[1] || dateStr.split('T')[1];
  return timePart && !timePart.startsWith('00:00:00');
};

// Standardize date format for Lightweight Charts (supports both Intraday and Daily Unix Timestamps)
const normalizeChartTime = (dateStr) => {
  if (!dateStr) return null;

  // Handle "YYYY-MM-DD HH:MM:SS" (Intraday 5m format)
  if (dateStr.includes(':')) {
    const [datePart, timePart] = dateStr.split(' ');
    const [year, month, day] = datePart.split('-').map(Number);
    const [hours, minutes, seconds] = timePart.split(':').map(Number);

    // Parse as UTC seconds to ensure consistent bucket math across timezones
    return Math.floor(Date.UTC(year, month - 1, day, hours, minutes, seconds) / 1000);
  }

  // Handle "YYYY-MM-DD" (Daily format)
  const [year, month, day] = dateStr.split('-').map(Number);
  return Math.floor(Date.UTC(year, month - 1, day) / 1000);
};

const calculateSimDTE = (expirationStr, currentSimDateStr) => {
  if (!expirationStr || !currentSimDateStr) return 0;

  const currISO = typeof currentSimDateStr === 'string' 
    ? currentSimDateStr.replace(' ', 'T') 
    : currentSimDateStr;
  
  const curr = new Date(currISO);

  const expDateOnly = typeof expirationStr === 'string' 
    ? expirationStr.split('T')[0].split(' ')[0] 
    : expirationStr;

  const exp = new Date(`${expDateOnly}T16:00:00`);

  const diffMs = exp.getTime() - curr.getTime();
  return diffMs / (1000 * 60 * 60 * 24);
};

export default function App() {
  const chartContainerRef = useRef(null);
  const seriesRef = useRef(null);
  const chartRef = useRef(null);
  const seriesTypeRef = useRef('line');

  // Dynamic Stock Registry State
  const [availableSymbols, setAvailableSymbols] = useState(['SPY', 'QQQ']);
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');

  // 1. New Timeframe Interval State (5m default)
  const [selectedInterval, setSelectedInterval] = useState(5); // 5 or 15
  const selectedIntervalRef = useRef(selectedInterval);
  useEffect(() => {
    selectedIntervalRef.current = selectedInterval;
  }, [selectedInterval]);

  const selectedSymbolRef = useRef(selectedSymbol);
  useEffect(() => {
    selectedSymbolRef.current = selectedSymbol;
  }, [selectedSymbol]);

  // Master historical buffers (rawTicksMap stores raw time/price feeds for dynamic aggregation)
  const historyMapRef = useRef({});
  const rawTicksMapRef = useRef({});
  const currentCandleMapRef = useRef({});

  // Dynamic Price & Date State Maps
  const [marketPrices, setMarketPrices] = useState({});
  const [priceChanges, setPriceChanges] = useState({});
  const [simulatedDates, setSimulatedDates] = useState({});

  const [logs, setLogs] = useState([]);
  const [openPositions, setOpenPositions] = useState([]);
  const [closedPositions, setClosedPositions] = useState([]);
  const [activePositionsTab, setActivePositionsTab] = useState('OPEN');
  const [accountBalance, setAccountBalance] = useState(10000.0);
  const [npcStats, setNpcStats] = useState({});

  const currentPrice = marketPrices[selectedSymbol] || 0;
  const priceChange = priceChanges[selectedSymbol] || 0;
  const simulatedDate = simulatedDates[selectedSymbol] || '';

  // Helper to re-aggregate raw tick data into 5m or 15m candles dynamically
  const rebuildCandleHistory = (sym, intervalMinutes) => {
    const ticks = rawTicksMapRef.current[sym] || [];
    if (ticks.length === 0) return [];

    const isDaily = intervalMinutes >= 1440; // 1 day = 1440 mins
    const candleMap = new Map();

    // 1. Group ticks into simulation timestamp buckets
    ticks.forEach(({ time, price }) => {
      let candleTime;

      if (isDaily) {
        // Bucket by UTC midnight for daily candles
        const date = new Date(time * 1000);
        candleTime = Math.floor(
          Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) / 1000
        );
      } else {
        // Bucket by interval seconds (e.g. 5m = 300s, 15m = 900s)
        const bucketInterval = intervalMinutes * 60;
        candleTime = Math.floor(time / bucketInterval) * bucketInterval;
      }

      if (!candleMap.has(candleTime)) {
        candleMap.set(candleTime, {
          time: candleTime,
          prices: [price],
        });
      } else {
        candleMap.get(candleTime).prices.push(price);
      }
    });

    // 2. Sort bucket keys chronologically
    const sortedTimes = Array.from(candleMap.keys()).sort((a, b) => a - b);
    const candles = [];
    let prevClose = null;

    // 3. Construct continuous candles
    sortedTimes.forEach((candleTime) => {
      const { prices } = candleMap.get(candleTime);
      const firstPrice = prices[0];
      const lastPrice = prices[prices.length - 1];

      const openPrice = prevClose !== null ? prevClose : firstPrice;
      const highPrice = Math.max(openPrice, ...prices);
      const lowPrice = Math.min(openPrice, ...prices);

      const candle = {
        time: candleTime,
        open: openPrice,
        high: highPrice,
        low: lowPrice,
        close: lastPrice,
      };

      candles.push(candle);
      prevClose = lastPrice;
    });

    currentCandleMapRef.current[sym] = candles[candles.length - 1] || null;
    return candles;
  };

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

  // Option Expiration Evaluator
  useEffect(() => {
    if (!openPositions || openPositions.length === 0) return;

    const remainingOpen = [];
    const newlyClosed = [];
    let totalPayout = 0;

    openPositions.forEach((pos) => {
      const activeDate = simulatedDates[pos.symbol] || simulatedDate;
      if (!activeDate) {
        remainingOpen.push(pos);
        return;
      }

      const dte = calculateSimDTE(pos.expiration, activeDate);
      const activePrice = marketPrices[pos.symbol] || currentPrice;

      if (dte < 0) {
        const intrinsicPrice = pos.type === 'CALL' 
          ? Math.max(0, activePrice - pos.strike)
          : Math.max(0, pos.strike - activePrice);

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
        remainingOpen.push({ ...pos, dte: Math.max(0, Math.ceil(dte)) });
      }
    });

    if (newlyClosed.length > 0) {
      setOpenPositions(remainingOpen);
      setClosedPositions((prev) => [...newlyClosed, ...prev]);
      setAccountBalance((prev) => prev + totalPayout);
    }
  }, [simulatedDates, marketPrices]);

  // Dynamic WebSocket Feed & Stock Registry Updates
  useEffect(() => {
    let isMounted = true;
    const ws = new WebSocket('ws://localhost:8000/ws/stocks');

    ws.onopen = () => {
      if (!isMounted) return ws.close();
      console.log('Connected to Multi-Stock WS Feed');
    };

    ws.onmessage = (event) => {
      if (!isMounted) return;

      try {
        const packet = JSON.parse(event.data);

        // 1. Dynamic Registry Update
        if (packet.dates) {
          setSimulatedDates(packet.dates);
          const incomingSymbols = Object.keys(packet.dates);
          if (incomingSymbols.length > 0) {
            setAvailableSymbols((prev) => Array.from(new Set([...prev, ...incomingSymbols])));
          }
        }

        // 2. Dynamic Price & Data Access Processing
        if (packet.data) {
          setMarketPrices((prev) => {
            const newChanges = {};
            Object.keys(packet.data).forEach((sym) => {
              const oldPrice = prev[sym] ?? packet.data[sym];
              newChanges[sym] = packet.data[sym] - oldPrice;
            });
            setPriceChanges(newChanges);
            return packet.data;
          });

          // Continuous Background Data Access & Chart Buffering
          Object.keys(packet.data).forEach((sym) => {
            const symPrice = packet.data[sym];
            const symDateStr = packet.dates?.[sym];
            if (symPrice === undefined || !symDateStr) return;

            if (!historyMapRef.current[sym]) historyMapRef.current[sym] = [];
            if (!rawTicksMapRef.current[sym]) rawTicksMapRef.current[sym] = [];

            const simDateStr = packet.dates[sym]; // e.g., "2026-08-26 09:30:00" or "2026-08-26"
            const timestampSec = normalizeChartTime(simDateStr);

            if (timestampSec !== null) {
              // Push the SIMULATED time in seconds into your raw tick buffer
              rawTicksMapRef.current[sym].push({ time: timestampSec, price: symPrice });
            }
            if (!timestampSec) return;

            const isIntraday = isIntradayDate(symDateStr);

            // Inside ws.onmessage -> Object.keys(packet.data).forEach((sym) => { ...

            if (!isIntraday) {
              // Line Series Mode
              const history = historyMapRef.current[sym] || [];
              const lastPoint = history[history.length - 1];
              const newPoint = { time: timestampSec, value: symPrice };

              if (!lastPoint || lastPoint.time < timestampSec) {
                history.push(newPoint);
              } else if (lastPoint.time === timestampSec) {
                history[history.length - 1] = newPoint;
              }
              historyMapRef.current[sym] = history;

              if (selectedSymbolRef.current === sym && seriesRef.current && seriesTypeRef.current === 'line') {
                seriesRef.current.update(newPoint);
              }
            } else {
              // 1. Append raw tick to buffer
              rawTicksMapRef.current[sym].push({ time: timestampSec, price: symPrice });

              // 2. Re-aggregate for active symbol & interval to maintain exact continuous structure
              if (selectedSymbolRef.current === sym) {
                const updatedCandles = rebuildCandleHistory(sym, selectedIntervalRef.current);
                historyMapRef.current[sym] = updatedCandles;

                const activeCandle = updatedCandles[updatedCandles.length - 1];
                if (seriesRef.current && seriesTypeRef.current === 'candlestick' && activeCandle) {
                  seriesRef.current.update(activeCandle);
                }
              }
            }
          });
        }

        if (packet.npcs) {
          setNpcStats(packet.npcs);
        }

        if (packet.logs && Array.isArray(packet.logs) && packet.logs.length > 0) {
          const formattedNewLogs = packet.logs.map((item, idx) => ({
            id: `log_ws_${Date.now()}_${idx}_${Math.random().toString(36).substr(2, 4)}`,
            text: typeof item === 'string' ? item : item.text,
            symbol: item.symbol || selectedSymbolRef.current,
            isDivider: item.isDivider || false
          }));

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
        ws.onopen = () => ws.close();
      }
    };
  }, []);

  // Sync Chart Rendering dynamically on symbol selection, interval, or date mode change
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
        secondsVisible: false,
      },
      localization: { dateFormat: 'yyyy-MM-dd' }
    });

    const activeDateStr = simulatedDates[selectedSymbol] || '';
    const isIntraday = isIntradayDate(activeDateStr);

    let series;
    if (!isIntraday) {
      seriesTypeRef.current = 'line';
      series = typeof chart.addAreaSeries === 'function'
        ? chart.addAreaSeries({
            topColor: 'rgba(38, 166, 154, 0.56)',
            bottomColor: 'rgba(38, 166, 154, 0.04)',
            lineColor: '#26a69a',
            lineWidth: 2,
          })
        : chart.addSeries(LineSeries, { color: '#26a69a', lineWidth: 2 });

      const cachedHistory = historyMapRef.current[selectedSymbol] || [];
      if (cachedHistory.length > 0) {
        series.setData(cachedHistory);
      }
    } else {
      seriesTypeRef.current = 'candlestick';
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

      // Hydrate dynamically using selected interval (5m vs 15m)
      const candleHistory = rebuildCandleHistory(selectedSymbol, selectedInterval);
      historyMapRef.current[selectedSymbol] = candleHistory;
      if (candleHistory.length > 0) {
        series.setData(candleHistory);
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
  }, [selectedSymbol, selectedInterval, isIntradayDate(simulatedDates[selectedSymbol])]);

  const activeNpc = npcStats[selectedSymbol] || { bull: {}, bear: {} };

  const filteredLogs = logs.filter(
    (log) => !log.symbol || log.symbol === selectedSymbol
  );

  const isIntradayActive = isIntradayDate(simulatedDates[selectedSymbol] || '');

  const dashboardView = (
    <>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid #1e293b', paddingBottom: '16px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            QuantWorld Sandbox
          </h1>
          <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: '14px' }}>Multi-Asset Modular Sandbox & RL Speculation</p>
        </div>

        {/* Dynamic Stock Registry Dropdown */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ color: '#94a3b8', fontSize: '13px', fontWeight: 'bold' }}>Asset:</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: '1px solid #334155',
              backgroundColor: '#0f172a',
              color: '#38bdf8',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '14px',
              outline: 'none',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            }}
          >
            {availableSymbols.map((symbol) => (
              <option key={symbol} value={symbol} style={{ backgroundColor: '#0f172a', color: '#f8fafc' }}>
                {symbol}
              </option>
            ))}
          </select>
        </div>

        {currentPrice !== 0 && (
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
        <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '16px', border: '1px solid #1e293b' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h2 style={{ fontSize: '16px', margin: 0, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
              {selectedSymbol} Real-Time Stream {isIntradayActive && `(${selectedInterval}m)`}
            </h2>

            {/* Interval Selector Controls (Visible only on Intraday Candlestick charts) */}
            {isIntradayActive && (
              <div style={{ display: 'flex', gap: '4px', backgroundColor: '#0f172a', padding: '2px', borderRadius: '6px', border: '1px solid #1e293b' }}>
                <button
                  onClick={() => setSelectedInterval(5)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '4px',
                    border: 'none',
                    backgroundColor: selectedInterval === 5 ? '#38bdf8' : 'transparent',
                    color: selectedInterval === 5 ? '#0f172a' : '#94a3b8',
                    fontWeight: 'bold',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  5m
                </button>
                <button
                  onClick={() => setSelectedInterval(15)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: '4px',
                    border: 'none',
                    backgroundColor: selectedInterval === 15 ? '#38bdf8' : 'transparent',
                    color: selectedInterval === 15 ? '#0f172a' : '#94a3b8',
                    fontWeight: 'bold',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  15m
                </button>
              </div>
            )}
          </div>
          <div ref={chartContainerRef} style={{ width: '100%' }} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #22c55e' }}>
              <div style={{ fontSize: '12px', color: '#22c55e', fontWeight: 'bold', textTransform: 'uppercase' }}>Bull Agent</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f8fafc', margin: '4px 0' }}>
                ${activeNpc.bull?.portfolio_value?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '10,000.00'}
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Cash: ${activeNpc.bull?.cash?.toFixed(2) || '10000.00'}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Pos: {activeNpc.bull?.contracts || 0} Contract(s)</div>
            </div>

            <div style={{ backgroundColor: '#131722', borderRadius: '8px', padding: '12px', border: '1px solid #1e293b', borderTop: '3px solid #ef4444' }}>
              <div style={{ fontSize: '12px', color: '#ef4444', fontWeight: 'bold', textTransform: 'uppercase' }}>Bear Agent</div>
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