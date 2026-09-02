// --- Options Chain Component with DTE / Expiration Selector ---
import React, { useState, useEffect, useCallback, useRef } from 'react';

export function OptionsChain({ 
  symbol = 'SPY',             // <--- NEW: Dynamic symbol prop (defaults to SPY)
  interval = '1d',            // <--- NEW: Dynamic interval prop (e.g., '5m', '15m', '1d')
  currentPrice, 
  simulatedDate, 
  onExecuteTrade 
}) {
  const [chain, setChain] = useState([]);
  const [selectedContract, setSelectedContract] = useState(null);
  const [contractsCount, setContractsCount] = useState(1);
  const [loading, setLoading] = useState(false);

  // Expiration / DTE state
  const [expirations, setExpirations] = useState([]);
  const [selectedExpiration, setSelectedExpiration] = useState('');

  // Use ref to synchronously lock user selection across state batches
  const userSelectedExpRef = useRef('');

  // Track initial load vs background tick refreshes
  const isInitialLoadRef = useRef(true);
  const activeFetchControllerRef = useRef(null);

  // Normalize simulatedDate to string or ISO format for API query
  // For intraday (5m ticks), we pass full ISO string or full date-time if available
  const simDay = simulatedDate ? String(simulatedDate).slice(0, 10) : '';
  const simDateTime = simulatedDate ? new Date(simulatedDate).toISOString() : '';

  // Parse simulatedDate or fall back to system today
  const getSimulatedToday = useCallback(() => {
    const today = simDay ? new Date(`${simDay}T00:00:00`) : new Date();
    today.setHours(0, 0, 0, 0);
    return today;
  }, [simDay]);

  // Calculate DTE relative to the SIMULATED date
  const calculateDTE = useCallback(
    (expDateStr) => {
      if (!expDateStr) return 0;
      const today = getSimulatedToday();
      const expDate = new Date(`${expDateStr}T00:00:00`);
      expDate.setHours(0, 0, 0, 0);
      const diffTime = expDate - today;
      return Math.max(-1, Math.ceil(diffTime / (1000 * 60 * 60 * 24)));
    },
    [getSimulatedToday]
  );

  // Generate fallback dates relative to the SIMULATED date
  const generateFallbackExpirations = useCallback(() => {
    const dates = [];
    const today = getSimulatedToday();
    [0, 1, 7, 14, 30, 45, 60].forEach((days) => {
      const d = new Date(today);
      d.setDate(d.getDate() + days);
      dates.push(d.toISOString().split('T')[0]);
    });
    return dates;
  }, [getSimulatedToday]);

  // Build URL query param safely
  const buildSimQuery = (dateStr) => {
    if (!dateStr) return '';
    return `&simulated_date=${encodeURIComponent(dateStr)}`;
  };

  // Reset selected contract if symbol or interval changes
  useEffect(() => {
    setSelectedContract(null);
    userSelectedExpRef.current = '';
    isInitialLoadRef.current = true;
  }, [symbol, interval]);

  // 1. Fetch available Expiration Dates for the specific symbol
  useEffect(() => {
    let isMounted = true;
    if (expirations.length === 0) {
      setLoading(true);
    }

    const simQuery = buildSimQuery(simDay);
    
    // NEW: Appended ?symbol= & interval= parameters
    const url = `http://localhost:8000/api/options-expirations?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${simQuery}`;

    fetch(url)
      .then((res) => res.json())
      .then((dates) => {
        if (!isMounted) return;
        const expList = Array.isArray(dates) ? dates : dates?.expirations || [];
        const validExps = expList.length > 0 ? expList : generateFallbackExpirations();

        setExpirations(validExps);

        const currentActive = userSelectedExpRef.current || selectedExpiration;

        if (currentActive) {
          const activeClean = currentActive.slice(0, 10);
          const matchedExp = validExps.find((e) => e.slice(0, 10) === activeClean);

          if (matchedExp) {
            setSelectedExpiration(matchedExp);
            return;
          } else {
            setSelectedExpiration(currentActive);
            return;
          }
        }

        const defaultExp = validExps[0] || '';
        setSelectedExpiration(defaultExp);
        userSelectedExpRef.current = defaultExp;
      })
      .catch(() => {
        if (!isMounted) return;
        const fallbackDates = generateFallbackExpirations();
        setExpirations(fallbackDates);

        const currentActive = userSelectedExpRef.current || selectedExpiration;
        const matched = fallbackDates.find((e) => e.slice(0, 10) === currentActive?.slice(0, 10));

        const fallbackExp = matched || currentActive || fallbackDates[0] || '';
        setSelectedExpiration(fallbackExp);
      })
      .finally(() => {
        if (isMounted && expirations.length === 0) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [symbol, interval, simDay]); // Refetch expirations if symbol, interval, or date changes

  // 2. Fetch Option Chain using symbol, interval & simulatedDate
  const fetchChain = useCallback((sym, tf, price, expDate, simDateStr, showLoader = false) => {
    if (price == null || isNaN(price) || !expDate) return;

    if (activeFetchControllerRef.current) {
      activeFetchControllerRef.current.abort();
    }
    const controller = new AbortController();
    activeFetchControllerRef.current = controller;

    if (showLoader) {
      setLoading(true);
    }

    const simQuery = buildSimQuery(simDateStr);
    
    // NEW: Include symbol & interval in options-chain query
    const url = `http://localhost:8000/api/options-chain?symbol=${encodeURIComponent(sym)}&interval=${encodeURIComponent(tf)}&current_price=${price}&expiration=${expDate}${simQuery}&_t=${Date.now()}`;

    fetch(url, { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => {
        const chainData = Array.isArray(data) ? data : data?.chain || [];

        setChain(chainData);

        setSelectedContract((prevSelected) => {
          if (!prevSelected) return null;
          const updatedRow = chainData.find((r) => r.strike === prevSelected.strike);
          if (!updatedRow) return prevSelected;

          const isCall = prevSelected.type === 'CALL';
          return {
            ...prevSelected,
            ask: isCall ? updatedRow.call_ask : updatedRow.put_ask,
            bid: isCall ? updatedRow.call_bid : updatedRow.put_bid,
          };
        });
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          console.error('Error fetching options chain:', err);
        }
      })
      .finally(() => {
        if (showLoader) {
          setLoading(false);
        }
      });
  }, []);

  useEffect(() => {
    if (selectedExpiration) {
      const showLoader = isInitialLoadRef.current;
      // For 5m interval, pass full simDateTime so the backend can evaluate intraday options pricing
      const targetSimDate = interval === '5m' ? simDateTime : simDay;

      fetchChain(symbol, interval, currentPrice, selectedExpiration, targetSimDate, showLoader);

      if (isInitialLoadRef.current) {
        isInitialLoadRef.current = false;
      }
    }
  }, [symbol, interval, currentPrice, selectedExpiration, simDay, simDateTime, fetchChain]);

  const handleExpirationChange = (e) => {
    const newExp = e.target.value;
    userSelectedExpRef.current = newExp;
    setSelectedExpiration(newExp);
    setSelectedContract(null);

    const targetSimDate = interval === '5m' ? simDateTime : simDay;
    fetchChain(symbol, interval, currentPrice, newExp, targetSimDate, true);
  };

  const handleTrade = () => {
    if (!selectedContract) return;

    const latestDTE = calculateDTE(selectedExpiration);

    if (latestDTE < 0) {
      alert(`Cannot execute order: Expiration date (${selectedExpiration}) has already passed!`);
      setSelectedContract(null);
      return;
    }

    const askPrice = selectedContract.ask ?? 0;
    onExecuteTrade({
      symbol,                    // <--- Included symbol in trade payload
      type: selectedContract.type,
      strike: selectedContract.strike,
      expiration: selectedExpiration,
      dte: latestDTE,
      premium: askPrice,
      bid: selectedContract.bid ?? 0,
      contracts: contractsCount,
      totalCost: askPrice * 100 * contractsCount,
    });

    setSelectedContract(null);
    setContractsCount(1);
  };

  const currentDTE = calculateDTE(selectedExpiration);

  return (
    <div
      style={{
        backgroundColor: '#131722',
        border: '1px solid #1e293b',
        borderRadius: '8px',
        padding: '20px',
        color: '#e2e8f0',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      {/* Expiration & DTE Top Control Strip */}
      <div
        style={{
          backgroundColor: '#0b0e14',
          border: '1px solid #1e293b',
          borderRadius: '6px',
          padding: '10px 16px',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          justify: 'space-between',
          width: '100%',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span
            style={{
              backgroundColor: '#38bdf8',
              color: '#0f172a',
              fontWeight: 'bold',
              fontSize: '11px',
              padding: '2px 6px',
              borderRadius: '4px',
              textTransform: 'uppercase',
            }}
          >
            {symbol} ({interval})
          </span>

          <label
            style={{
              fontSize: '12px',
              color: '#94a3b8',
              fontWeight: 'bold',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            Expiration Date:
          </label>
          <select
            value={selectedExpiration}
            onChange={handleExpirationChange}
            style={{
              backgroundColor: '#131722',
              color: '#38bdf8',
              border: '1px solid #38bdf8',
              borderRadius: '4px',
              padding: '6px 12px',
              fontSize: '13px',
              fontWeight: 'bold',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            {expirations.map((exp) => {
              const dte = calculateDTE(exp);
              return (
                <option key={exp} value={exp}>
                  {exp} ({dte}d)
                </option>
              );
            })}
          </select>
        </div>

        {/* Pushed to the far right */}
        <div
          style={{
            fontSize: '13px',
            color: '#e2e8f0',
            backgroundColor: '#1e293b',
            padding: '4px 12px',
            borderRadius: '4px',
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
            marginLeft: 'auto',
          }}
        >
          <span style={{ color: '#f8fafc', fontWeight: 'bold' }}>
            {selectedExpiration || simDay || '---'}
          </span>
          <span style={{ color: '#64748b' }}>•</span>
          <span>
            DTE: <strong style={{ color: '#38bdf8' }}>{currentDTE} Days</strong>
          </span>
        </div>
      </div>

      {/* Options Chain Table */}
      <div
        style={{
          maxHeight: '550px',
          overflowY: 'auto',
          border: '1px solid #1e293b',
          borderRadius: '4px',
        }}
      >
        {loading ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#94a3b8' }}>
            Loading {symbol} Options Chain...
          </div>
        ) : chain.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#ef4444' }}>
            No options data returned for {symbol} ({selectedExpiration}). Check backend endpoint at
            http://localhost:8000/api/options-chain.
          </div>
        ) : (
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '12px',
              textAlign: 'center',
            }}
          >
            <thead>
              <tr
                style={{
                  backgroundColor: '#0f172a',
                  color: '#94a3b8',
                  borderBottom: '1px solid #1e293b',
                }}
              >
                <th colSpan="2" style={{ padding: '8px', color: '#22c55e', borderRight: '1px solid #1e293b' }}>
                  CALLS
                </th>
                <th style={{ padding: '8px' }}>STRIKE</th>
                <th colSpan="2" style={{ padding: '8px', color: '#ef4444', borderLeft: '1px solid #1e293b' }}>
                  PUTS
                </th>
              </tr>
              <tr
                style={{
                  backgroundColor: '#0b0e14',
                  color: '#64748b',
                  fontSize: '11px',
                  borderBottom: '1px solid #1e293b',
                }}
              >
                <th style={{ padding: '4px' }}>BID</th>
                <th style={{ padding: '4px', borderRight: '1px solid #1e293b' }}>ASK</th>
                <th style={{ padding: '4px' }}>STRIKE</th>
                <th style={{ padding: '4px', borderLeft: '1px solid #1e293b' }}>BID</th>
                <th style={{ padding: '4px' }}>ASK</th>
              </tr>
            </thead>
            <tbody>
              {chain.map((row) => {
                const isSelectedCall =
                  selectedContract?.strike === row.strike && selectedContract?.type === 'CALL';
                const isSelectedPut =
                  selectedContract?.strike === row.strike && selectedContract?.type === 'PUT';

                return (
                  <tr
                    key={row.strike}
                    style={{ borderBottom: '1px solid #1e293b', backgroundColor: '#131722' }}
                  >
                    {/* Call Bid / Ask */}
                    <td
                      style={{
                        padding: '6px',
                        backgroundColor: row.in_the_money_call
                          ? 'rgba(34, 197, 94, 0.08)'
                          : 'transparent',
                        color: '#94a3b8',
                      }}
                    >
                      ${row.call_bid != null ? Number(row.call_bid).toFixed(2) : '---'}
                    </td>
                    <td
                      style={{
                        padding: '6px',
                        borderRight: '1px solid #1e293b',
                        backgroundColor: row.in_the_money_call
                          ? 'rgba(34, 197, 94, 0.08)'
                          : 'transparent',
                      }}
                    >
                      <button
                        onClick={() =>
                          setSelectedContract({
                            strike: row.strike,
                            type: 'CALL',
                            ask: row.call_ask,
                            bid: row.call_bid,
                          })
                        }
                        style={{
                          width: '100%',
                          padding: '4px 6px',
                          borderRadius: '4px',
                          border: isSelectedCall ? '1px solid #22c55e' : '1px solid #1e293b',
                          backgroundColor: isSelectedCall
                            ? 'rgba(34, 197, 94, 0.25)'
                            : '#0b0e14',
                          color: '#22c55e',
                          fontWeight: isSelectedCall ? 'bold' : 'normal',
                          cursor: 'pointer',
                          fontSize: '12px',
                        }}
                      >
                        ${row.call_ask != null ? Number(row.call_ask).toFixed(2) : '---'}
                      </button>
                    </td>

                    {/* Strike Price */}
                    <td
                      style={{
                        padding: '6px',
                        fontWeight: 'bold',
                        color: '#f8fafc',
                        backgroundColor: '#0f172a',
                      }}
                    >
                      ${row.strike?.toFixed(2)}
                    </td>

                    {/* Put Bid / Ask */}
                    <td
                      style={{
                        padding: '6px',
                        borderLeft: '1px solid #1e293b',
                        backgroundColor: row.in_the_money_put
                          ? 'rgba(239, 68, 68, 0.08)'
                          : 'transparent',
                        color: '#94a3b8',
                      }}
                    >
                      ${row.put_bid != null ? Number(row.put_bid).toFixed(2) : '---'}
                    </td>
                    <td
                      style={{
                        padding: '6px',
                        backgroundColor: row.in_the_money_put
                          ? 'rgba(239, 68, 68, 0.08)'
                          : 'transparent',
                      }}
                    >
                      <button
                        onClick={() =>
                          setSelectedContract({
                            strike: row.strike,
                            type: 'PUT',
                            ask: row.put_ask,
                            bid: row.put_bid,
                          })
                        }
                        style={{
                          width: '100%',
                          padding: '4px 6px',
                          borderRadius: '4px',
                          border: isSelectedPut ? '1px solid #ef4444' : '1px solid #1e293b',
                          backgroundColor: isSelectedPut ? 'rgba(239, 68, 68, 0.25)' : '#0b0e14',
                          color: '#ef4444',
                          fontWeight: isSelectedPut ? 'bold' : 'normal',
                          cursor: 'pointer',
                          fontSize: '12px',
                        }}
                      >
                        ${row.put_ask != null ? Number(row.put_ask).toFixed(2) : '---'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Order Summary & Execution Bar */}
      {selectedContract ? (
        <div
          style={{
            marginTop: '16px',
            padding: '12px 16px',
            backgroundColor: '#0b0e14',
            borderRadius: '6px',
            border: '1px solid #1e293b',
            display: 'flex',
            justify: 'space-between',
            alignItems: 'center',
            width: '100%',
            boxSizing: 'border-box',
          }}
        >
          {/* Order Text (Left Side) */}
          <div>
            <span
              style={{
                fontSize: '11px',
                color: '#94a3b8',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Order Summary
            </span>
            <div
              style={{
                fontWeight: 'bold',
                fontSize: '14px',
                color: selectedContract.type === 'CALL' ? '#22c55e' : '#ef4444',
                marginTop: '2px',
              }}
            >
              Buy {contractsCount}x {symbol} ${selectedContract.strike} {selectedContract.type} (
              {selectedExpiration} • {currentDTE}d)
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
              Ask: ${selectedContract.ask != null ? Number(selectedContract.ask).toFixed(2) : '0.00'} |
              Bid: ${selectedContract.bid != null ? Number(selectedContract.bid).toFixed(2) : '0.00'} |
              Total Cost:{' '}
              <strong style={{ color: '#e2e8f0' }}>
                ${((selectedContract.ask ?? 0) * 100 * contractsCount).toFixed(2)}
              </strong>
            </div>
          </div>

          {/* Controls Pushed to Right Side */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginLeft: 'auto' }}>
            <label
              style={{
                fontSize: '12px',
                color: '#94a3b8',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              Qty:
              <input
                type="number"
                min="1"
                max="50"
                value={contractsCount}
                onChange={(e) =>
                  setContractsCount(Math.max(1, parseInt(e.target.value, 10) || 1))
                }
                style={{
                  width: '55px',
                  backgroundColor: '#131722',
                  border: '1px solid #1e293b',
                  color: '#fff',
                  borderRadius: '4px',
                  padding: '4px 6px',
                  textAlign: 'center',
                }}
              />
            </label>

            <button
              onClick={handleTrade}
              disabled={currentDTE < 0}
              style={{
                padding: '8px 18px',
                backgroundColor: currentDTE < 0 ? '#475569' : '#38bdf8',
                color: currentDTE < 0 ? '#94a3b8' : '#0f172a',
                fontWeight: 'bold',
                border: 'none',
                borderRadius: '4px',
                cursor: currentDTE < 0 ? 'not-allowed' : 'pointer',
              }}
            >
              {currentDTE < 0 ? 'Option Expired' : 'Execute Order'}
            </button>
          </div>
        </div>
      ) : (
        <div
          style={{
            marginTop: '16px',
            padding: '12px',
            textAlign: 'center',
            color: '#64748b',
            fontSize: '13px',
            backgroundColor: '#0b0e14',
            borderRadius: '6px',
            border: '1px dotted #1e293b',
          }}
        >
          Click an <strong style={{ color: '#22c55e' }}>Ask</strong> price to buy a Call, or a{' '}
          <strong style={{ color: '#ef4444' }}>Put Ask</strong> to buy a Put.
        </div>
      )}
    </div>
  );
}

// --- Positions & Active Orders Table ---
export function PositionsTable({ positions, currentPrice, onClosePosition }) {
  if (!positions || positions.length === 0) {
    return (
      <div style={{ backgroundColor: '#131722', border: '1px solid #1e293b', borderRadius: '8px', padding: '20px', color: '#64748b', textAlign: 'center', fontSize: '13px' }}>
        No open positions. Use the Options Chain to execute a trade.
      </div>
    );
  }

  // Simple Black-Scholes estimate for live position valuation on price ticks
  const calculateLiveContractValue = (pos) => {
    const spot = currentPrice ?? pos.strike;
    const strike = pos.strike;
    const dte = Math.max(pos.dte ?? 1, 0.5);
    const iv = 0.20;

    const callIntrinsic = Math.max(0, spot - strike);
    const putIntrinsic = Math.max(0, strike - spot);
    const intrinsic = pos.type === 'CALL' ? callIntrinsic : putIntrinsic;

    const t = dte / 365.0;
    const timeValue = spot * iv * Math.sqrt(t) * 0.4 * Math.exp(-Math.pow((spot - strike) / (spot * 0.05), 2));
    
    // Estimated current mid/mark price per share
    const markPrice = Math.max(0.01, intrinsic + timeValue);
    return markPrice;
  };

  return (
    <div style={{ backgroundColor: '#131722', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', color: '#e2e8f0', marginTop: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4 style={{ margin: 0, fontSize: '15px', color: '#f8fafc' }}>Open Positions</h4>
        <span style={{ fontSize: '12px', color: '#94a3b8' }}>
          Active Contracts: <strong>{positions.length}</strong>
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: '#0f172a', color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
              <th style={{ padding: '8px 10px' }}>Type</th>
              <th style={{ padding: '8px 10px' }}>Strike</th>
              <th style={{ padding: '8px 10px' }}>Exp / DTE</th>
              <th style={{ padding: '8px 10px' }}>Qty</th>
              <th style={{ padding: '8px 10px' }}>Entry Premium</th>
              <th style={{ padding: '8px 10px' }}>Current Mark</th>
              <th style={{ padding: '8px 10px' }}>Unrealized P&L</th>
              <th style={{ padding: '8px 10px', textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos, idx) => {
              const liveMark = calculateLiveContractValue(pos);
              const totalCost = (pos.premium ?? 0) * 100 * (pos.contracts ?? 1);
              const currentValue = liveMark * 100 * (pos.contracts ?? 1);
              const pnl = currentValue - totalCost;
              const pnlPercent = totalCost > 0 ? (pnl / totalCost) * 100 : 0;
              const isProfit = pnl >= 0;

              return (
                <tr key={pos.id || `${pos.strike}-${pos.type}-${pos.expiration}-${idx}`} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '10px', fontWeight: 'bold', color: pos.type === 'CALL' ? '#22c55e' : '#ef4444' }}>
                    {pos.type}
                  </td>
                  <td style={{ padding: '10px', fontWeight: 'bold' }}>${Number(pos.strike).toFixed(2)}</td>
                  <td style={{ padding: '10px', color: '#94a3b8' }}>
                    {pos.expiration || 'Weekly'} ({pos.dte ?? 7}d)
                  </td>
                  <td style={{ padding: '10px' }}>{pos.contracts}x</td>
                  <td style={{ padding: '10px' }}>${Number(pos.premium).toFixed(2)}</td>
                  <td style={{ padding: '10px', color: '#38bdf8', fontWeight: 'bold' }}>
                    ${liveMark.toFixed(2)}
                  </td>
                  <td style={{ padding: '10px', fontWeight: 'bold', color: isProfit ? '#22c55e' : '#ef4444' }}>
                    {isProfit ? '+' : ''}${pnl.toFixed(2)} ({isProfit ? '+' : ''}{pnlPercent.toFixed(1)}%)
                  </td>
                  <td style={{ padding: '10px', textAlign: 'right' }}>
                    <button
                      onClick={() => onClosePosition(pos.id, liveMark)}
                      style={{
                        backgroundColor: '#ef4444',
                        color: '#ffffff',
                        border: 'none',
                        borderRadius: '4px',
                        padding: '5px 10px',
                        fontSize: '11px',
                        fontWeight: 'bold',
                        cursor: 'pointer'
                      }}
                    >
                      Close Position
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
