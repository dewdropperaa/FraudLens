import React, { useEffect, useState } from 'react';
import { Icons } from '../components';

const STREAMING_MESSAGES = {
  IdentityAgent: ["Verifying CIN format...", "Cross-referencing names...", "Checking identity reuse..."],
  DocumentAgent: ["Scanning document metadata...", "Checking for tampering...", "Validating stamps..."],
  PolicyAgent: ["Checking CNSS limits...", "Verifying coverage dates...", "Analyzing threshold gaming..."],
  AnomalyAgent: ["Analyzing behavioral patterns...", "Checking amount anomalies...", "Evaluating history..."],
  PatternAgent: ["Detecting scripted signatures...", "Checking timing patterns...", "Comparing past fraud..."],
  GraphRiskAgent: ["Building provider network...", "Analyzing clusters...", "Checking network risk..."],
  Consensus: ["Gathering agent votes...", "Calculating trust score...", "Finalizing decision..."],
  HumanReview: ["Awaiting human input...", "Reviewing flags...", "Pending final sign-off..."]
};

function StreamingReasoning({ agent }) {
  const [text, setText] = useState("");

  useEffect(() => {
    const messages = STREAMING_MESSAGES[agent] || ["Analyzing data..."];
    let isMounted = true;
    let msgIndex = 0;
    let charIndex = 0;
    let currentText = "";
    let timeoutId;

    const tick = () => {
      if (!isMounted) return;
      const msg = messages[msgIndex];
      if (charIndex < msg.length) {
        currentText += msg[charIndex];
        setText(currentText);
        charIndex++;
        timeoutId = setTimeout(tick, 50 + Math.random() * 50); // 50-100ms
      } else {
        timeoutId = setTimeout(() => {
          if (!isMounted) return;
          msgIndex = (msgIndex + 1) % messages.length;
          charIndex = 0;
          currentText = "";
          setText("");
          tick();
        }, 1500);
      }
    };

    timeoutId = setTimeout(tick, 100);
    
    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
    };
  }, [agent]);

  return <span>{text}<span className="animate-pulse">|</span></span>;
}

const AGENT_SEQUENCE = [
  "IdentityAgent",
  "DocumentAgent",
  "PolicyAgent",
  "AnomalyAgent",
  "PatternAgent",
  "GraphRiskAgent",
  "Consensus",
  "HumanReview"
];

const STATUS_COLORS = {
  PENDING: 'bg-gray-200 text-gray-500',
  RUNNING: 'bg-blue-500 text-white animate-pulse shadow-[0_0_15px_rgba(59,130,246,0.5)]',
  COMPLETED: 'bg-green-500 text-white transition-colors duration-500',
  FAILED: 'bg-red-500 text-white animate-pulse',
  SKIPPED: 'bg-gray-100 text-gray-400 border border-gray-300'
};

const STATUS_ICONS = {
  PENDING: () => <div className="w-4 h-4 rounded-full bg-yellow-400" />,
  RUNNING: () => null, // Handled inside the node
  COMPLETED: () => <div className="w-4 h-4 rounded-full bg-green-500" />,
  FAILED: () => <div className="w-4 h-4 rounded-full bg-red-500" />,
  SKIPPED: () => <div className="w-4 h-4 rounded-full bg-gray-300" />
};

export function ClaimFlowGraph({ claimId, agentOutputs = [] }) {
  const [flowState, setFlowState] = useState(null);

  useEffect(() => {
    if (!claimId) return;

    const fetchFlow = async () => {
      try {
        const token = localStorage.getItem('cg_token');
        const apiKey = import.meta.env.VITE_CLAIMAGUARD_API_KEY;
        const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
        const headers = token
          ? { 'Authorization': `Bearer ${token}` }
          : apiKey ? { 'X-API-Key': apiKey } : {};
        const res = await fetch(`${baseUrl}/claim/${claimId}/flow`, { headers });
        if (res.ok) {
          const data = await res.json();
          setFlowState(data.steps);
        }
      } catch (err) {
        console.error("Failed to fetch flow state", err);
      }
    };

    fetchFlow();
    const interval = setInterval(fetchFlow, 3000);
    return () => clearInterval(interval);
  }, [claimId]);

  if (!flowState) {
    return <div className="p-4 text-gray-500">Loading flow state...</div>;
  }

  const completedCount = Object.values(flowState).filter(s => s?.status === 'COMPLETED').length;
  const totalCount = AGENT_SEQUENCE.length;
  const progressPercent = Math.round((completedCount / totalCount) * 100);

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Icons.Activity /> Real-time Claim Flow
        </h3>
        <div className="text-sm text-gray-500">
          Progress: {completedCount} / {totalCount}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-8">
        <div 
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
          style={{ width: `${progressPercent}%` }}
        ></div>
      </div>

      {/* Flow Nodes */}
      <div className="flex items-center justify-between relative overflow-x-auto pb-8 pt-4 px-4">
        {/* Connecting Line */}
        <div className="absolute top-1/2 left-8 right-8 h-1 bg-gray-200 -z-10 -translate-y-1/2"></div>

        {AGENT_SEQUENCE.map((agent, index) => {
          const stateObj = flowState[agent] || { status: 'PENDING' };
          const status = typeof stateObj === 'string' ? stateObj : stateObj.status;
          const isHumanReview = agent === 'HumanReview';
          const isAmber = isHumanReview && status === 'RUNNING';
          const isFraud = stateObj.is_fraud;
          
          const hasRunningNode = Object.values(flowState).some(s => (typeof s === 'string' ? s : s.status) === 'RUNNING');
          const isDimmed = hasRunningNode && status !== 'RUNNING';
          
          let nodeClass = 'w-12 h-12 rounded-full flex items-center justify-center text-xs font-bold shadow-md transition-all duration-500 z-10 ';
          
          if (isAmber) {
            nodeClass += 'bg-amber-500 text-white animate-pulse ring-4 ring-amber-200';
          } else if (isFraud) {
            nodeClass += 'bg-red-600 text-white shadow-[0_0_15px_rgba(220,38,38,0.7)]';
          } else {
            nodeClass += STATUS_COLORS[status] || STATUS_COLORS.PENDING;
          }

          if (status === 'FAILED') {
            nodeClass += ' animate-[shake_0.5s_ease-in-out]';
          }

          const output = agentOutputs.find(o => o.agent === agent || o.agent_name === agent || o.agent_name?.replace(' ', '') === agent) || stateObj;

          return (
            <div key={agent} className={`flex flex-col items-center group relative cursor-help transition-opacity duration-500 ${status === 'SKIPPED' ? 'opacity-40 pointer-events-none' : isDimmed ? 'opacity-60' : 'opacity-100'}`}>
              <div className={nodeClass}>
                {status === 'RUNNING' && !isHumanReview ? (
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : status === 'COMPLETED' ? (
                  <Icons.CheckCircle className="w-5 h-5" />
                ) : status === 'FAILED' || isFraud ? (
                  <Icons.AlertTriangle className="w-5 h-5" />
                ) : (
                  index + 1
                )}
              </div>
              <div className="mt-2 text-xs font-medium text-gray-600 text-center w-20">
                {agent.replace('Agent', '')}
              </div>
              <div className="mt-1">
                {status === 'RUNNING' ? (
                  <span className="text-[10px] text-blue-500 animate-pulse font-medium">thinking...</span>
                ) : (
                  STATUS_ICONS[status]?.() || STATUS_ICONS.PENDING()
                )}
              </div>

              {/* Tooltip */}
              <div className="absolute bottom-full mb-2 hidden group-hover:block w-56 bg-white text-gray-800 text-xs rounded-lg p-3 shadow-xl border border-gray-200 z-20 transition-opacity duration-200">
                <div className="font-bold mb-2 border-b border-gray-100 pb-2 flex items-center justify-between">
                  <span>{agent}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] ${
                    status === 'COMPLETED' ? 'bg-green-100 text-green-700' :
                    status === 'RUNNING' ? 'bg-blue-100 text-blue-700' :
                    status === 'FAILED' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {status}
                  </span>
                </div>
                
                {isFraud && (
                  <div className="mb-2 text-red-600 font-semibold flex items-center gap-1">
                    <Icons.AlertTriangle className="w-3 h-3" /> ⚠️ Risk detected
                  </div>
                )}

                {status === 'RUNNING' && (
                  <div className="italic text-gray-500 mb-1">
                    <StreamingReasoning agent={agent} />
                  </div>
                )}

                {status === 'FAILED' && (
                  <div className="mb-2 text-red-600 text-xs bg-red-50 p-2 rounded border border-red-100">
                    <div className="font-semibold">Execution Failed</div>
                    <div className="mt-1 line-clamp-2 hover:line-clamp-none transition-all duration-300 cursor-pointer">
                      {output?.explanation || output?.reasoning || "An unexpected error occurred during agent execution."}
                    </div>
                  </div>
                )}

                {output && status !== 'FAILED' && (output.score !== undefined || output.confidence !== undefined) && (
                  <div className="space-y-1 mb-2">
                    {output.score !== undefined && output.score !== null && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Score:</span>
                        <span className="font-medium">{Number(output.score).toFixed(2)}</span>
                      </div>
                    )}
                    {output.confidence !== undefined && output.confidence !== null && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Confidence:</span>
                        <span className="font-medium">{Number(output.confidence).toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                )}
                
                {(output?.explanation || output?.reasoning) && status !== 'FAILED' && (
                  <div className="mt-2 text-gray-600 border-t border-gray-100 pt-2 line-clamp-3 hover:line-clamp-none transition-all duration-300 cursor-pointer">
                    {output.explanation || output.reasoning}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}