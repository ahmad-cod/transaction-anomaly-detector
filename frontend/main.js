// ── Data engine ──────────────────────────────
let allTxns = [];
let selectedId = null;
let activeFilter = "ALL";
let sortKey = "score";
let sortDesc = true;
let streamInterval = null;
let txnCounter = 0;
let streaming = true;

const RULE_LABELS = {
  "amt>$5k": { label: "Amount > $5,000", color: "#e8453c" },
  "micro-txn": { label: "Micro-transaction (card test)", color: "#e89a2a" },
  "night+foreign": { label: "Late-night foreign txn", color: "#e8453c" },
  velocity: { label: "High transaction velocity", color: "#e89a2a" },
  declines: { label: "Multiple prior declines", color: "#e89a2a" },
  "risk+night": { label: "High-risk merchant, night", color: "#e89a2a" },
};

function rng(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function generateTxn(seed) {
  const r = rng(seed);
  const isFraud = r() < 0.12;
  let amt, hr, freq1h, freq24h, cm, mr, dec;
  if (isFraud) {
    const t = Math.floor(r() * 3);
    if (t === 0) {
      amt = 3000 + r() * 12000;
      hr = Math.floor(r() * 4);
      freq1h = 10 + Math.floor(r() * 20);
      freq24h = 25 + Math.floor(r() * 35);
      cm = r() < 0.8 ? 0 : 1;
      mr = r() < 0.6 ? 2 : 1;
      dec = Math.floor(r() * 4) + 1;
    } else if (t === 1) {
      amt = Math.round(r() * 100) / 100 + 0.01;
      hr = Math.floor(r() * 5);
      freq1h = 15 + Math.floor(r() * 15);
      freq24h = 30 + Math.floor(r() * 30);
      cm = r() < 0.75 ? 0 : 1;
      mr = 2;
      dec = Math.floor(r() * 5) + 1;
    } else {
      amt = 200 + r() * 400;
      hr = 1 + Math.floor(r() * 4);
      freq1h = 8 + Math.floor(r() * 12);
      freq24h = 20 + Math.floor(r() * 25);
      cm = r() < 0.7 ? 0 : 1;
      mr = r() < 0.4 ? 1 : 2;
      dec = Math.floor(r() * 3) + 1;
    }
  } else {
    amt = Math.min(500, Math.max(5, Math.exp(3 + r() * 2.5)));
    hr = Math.floor(r() * 24);
    freq1h = Math.floor(r() * 4) + 1;
    freq24h = Math.floor(r() * 8) + 1;
    cm = r() > 0.05 ? 1 : 0;
    mr = r() < 0.7 ? 0 : r() < 0.92 ? 1 : 2;
    dec = Math.floor(r() * 2);
  }

  // Compute anomaly score
  const logAmt = Math.log1p(amt) / Math.log1p(15000);
  const isNight = hr < 6 || hr >= 22 ? 1 : 0;
  const velRatio = freq1h / Math.max(1, freq24h);
  const riskRaw = (mr * 2 + dec + (1 - cm) * 3 + isNight) / 12;
  const noise = (r() - 0.5) * 0.06;
  const score = Math.min(
    0.99,
    Math.max(
      0.01,
      logAmt * 0.3 + velRatio * 0.25 + riskRaw * 0.35 + isNight * 0.1 + noise,
    ),
  );

  const rules = [];
  if (amt > 5000) rules.push("amt>$5k");
  if (amt < 0.1) rules.push("micro-txn");
  if (isNight && !cm) rules.push("night+foreign");
  if (freq1h > 10) rules.push("velocity");
  if (dec > 2) rules.push("declines");
  if (mr === 2 && isNight) rules.push("risk+night");

  const decision = isFraud || score > 0.45 ? "FRAUD" : "NORMAL";
  const risk = score > 0.6 ? "HIGH" : score > 0.35 ? "MEDIUM" : "LOW";

  return {
    id: `TXN-${String(++txnCounter).padStart(6, "0")}`,
    amount: Math.round(amt * 100) / 100,
    hour: hr,
    freq1h,
    freq24h,
    cm,
    mr,
    dec,
    score: Math.round(score * 1000) / 1000,
    rules,
    decision,
    risk,
    isNew: true,
    ts: new Date(),
  };
}

// ── Stream ───────────────────────────────────
function startStream() {
  streamInterval = setInterval(() => {
    const t = generateTxn(Date.now() + Math.random() * 999999);
    allTxns.unshift(t);
    if (allTxns.length > 500) allTxns.pop();
    updateMetrics();
    renderTable();
    setTimeout(() => {
      t.isNew = false;
    }, 600);
  }, 1400);
}

function toggleStream() {
  const btn = document.getElementById("streamBtn");
  if (streaming) {
    clearInterval(streamInterval);
    streaming = false;
    btn.textContent = "▶ Resume";
    btn.classList.add("paused");
    document.getElementById("streamStatus").textContent = "Stream paused";
  } else {
    startStream();
    streaming = true;
    btn.textContent = "⏸ Pause";
    btn.classList.remove("paused");
    document.getElementById("streamStatus").textContent =
      "Streaming transactions…";
  }
}

// ── Metrics ──────────────────────────────────
function updateMetrics() {
  const total = allTxns.length;
  const fraud = allTxns.filter((t) => t.decision === "FRAUD").length;
  const high = allTxns.filter((t) => t.risk === "HIGH").length;
  const med = allTxns.filter((t) => t.risk === "MEDIUM").length;
  const avg = total ? allTxns.reduce((s, t) => s + t.score, 0) / total : 0;
  document.getElementById("mTotal").textContent = total;
  document.getElementById("mFraud").textContent = fraud;
  document.getElementById("mFraudPct").textContent = total
    ? `${Math.round((fraud / total) * 100)}% of total`
    : "0% of total";
  document.getElementById("mHigh").textContent = high;
  document.getElementById("mMed").textContent = med;
  document.getElementById("mAvg").textContent = avg.toFixed(3);
}

// ── Filter / sort ────────────────────────────
function setFilter(f, btn) {
  activeFilter = f;
  document.querySelectorAll(".filter-btn").forEach((b) => {
    b.className = "filter-btn";
    const map = {
      ALL: "active-all",
      HIGH: "active-high",
      MEDIUM: "active-medium",
      LOW: "active-low",
      FRAUD: "active-high",
    };
    if (b.dataset.filter === f) b.classList.add(map[f] || "active-all");
  });
  renderTable();
}

const sortCycle = [
  ["score", true],
  ["amount", true],
  ["hour", false],
];
let sortCycleIdx = 0;
function cycleSort() {
  sortCycleIdx = (sortCycleIdx + 1) % sortCycle.length;
  [sortKey, sortDesc] = sortCycle[sortCycleIdx];
  const labels = { score: "Score ↓", amount: "Amount ↓", hour: "Time ↑" };
  document.getElementById("sortLabel").textContent = labels[sortKey];
  renderTable();
}
function sortBy(k) {
  sortDesc = sortKey === k ? !sortDesc : true;
  sortKey = k;
  renderTable();
}

function filteredTxns() {
  const q = document.getElementById("searchBox").value.toLowerCase();
  return allTxns
    .filter((t) => {
      if (activeFilter === "HIGH") return t.risk === "HIGH";
      if (activeFilter === "MEDIUM") return t.risk === "MEDIUM";
      if (activeFilter === "LOW") return t.risk === "LOW";
      if (activeFilter === "FRAUD") return t.decision === "FRAUD";
      return true;
    })
    .filter(
      (t) =>
        !q ||
        t.id.toLowerCase().includes(q) ||
        String(t.amount).includes(q) ||
        t.rules.join(" ").includes(q),
    )
    .sort((a, b) => {
      const av =
        sortKey === "amount" ? a.amount : sortKey === "hour" ? a.hour : a.score;
      const bv =
        sortKey === "amount" ? b.amount : sortKey === "hour" ? b.hour : b.score;
      return sortDesc ? bv - av : av - bv;
    });
}

// ── Table render ─────────────────────────────
function scoreColor(s) {
  return s > 0.6 ? "var(--red)" : s > 0.35 ? "var(--amber)" : "var(--green)";
}

function renderTable() {
  const rows = filteredTxns();
  const body = document.getElementById("tableBody");
  const empty = document.getElementById("emptyState");
  if (!rows.length) {
    body.innerHTML = "";
    empty.style.display = "flex";
    return;
  }
  empty.style.display = "none";

  body.innerHTML = rows
    .slice(0, 120)
    .map((t) => {
      const sc = scoreColor(t.score);
      const chips = t.rules
        .slice(0, 2)
        .map((r) => `<span class="rule-chip">${r}</span>`)
        .join("");
      const more =
        t.rules.length > 2
          ? `<span class="rule-chip">+${t.rules.length - 2}</span>`
          : "";
      return `<tr onclick="selectTxn('${t.id}')" class="${t.isNew ? "new-row" : ""} ${selectedId === t.id ? "selected" : ""}">
      <td><span class="txn-id">${t.id}</span></td>
      <td><span class="amount">$${t.amount.toFixed(2)}</span></td>
      <td><span class="hour-cell">${String(t.hour).padStart(2, "0")}:00</span></td>
      <td>
        <div class="score-cell">
          <span class="score-num" style="color:${sc}">${t.score.toFixed(3)}</span>
          <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${t.score * 100}%;background:${sc}"></div></div>
        </div>
      </td>
      <td><span class="risk-badge risk-${t.risk}">${t.risk}</span></td>
      <td><span class="decision-badge dec-${t.decision}">${t.decision}</span></td>
      <td>${chips}${more}${!t.rules.length ? '<span style="color:var(--text2);font-size:11px;font-family:var(--mono)">ML only</span>' : ""}</td>
    </tr>`;
    })
    .join("");
}

// ── Detail panel ─────────────────────────────
let sparkChart = null;

function selectTxn(id) {
  selectedId = id;
  renderTable();
  const t = allTxns.find((x) => x.id === id);
  if (!t) return;

  document.getElementById("panelEmpty").style.display = "none";
  document.getElementById("panelBody").style.display = "block";

  const sc = scoreColor(t.score);
  const ruleItems = t.rules.length
    ? t.rules
        .map((r) => {
          const info = RULE_LABELS[r] || { label: r, color: "var(--text1)" };
          return `<div class="rule-item"><span class="rule-dot" style="background:${info.color}"></span>${info.label}</div>`;
        })
        .join("")
    : `<div class="rule-item"><span class="rule-dot" style="background:var(--blue)"></span>ML model detection only</div>`;

  const ctx_id = "sparkCanvas_" + id;

  document.getElementById("panelBody").innerHTML = `
    <div class="panel-header">
      <div>
        <div class="panel-title">Transaction detail</div>
        <div class="panel-amount">$${t.amount.toFixed(2)}</div>
        <div style="margin-top:4px"><span class="risk-badge risk-${t.risk}">${t.risk}</span> <span class="decision-badge dec-${t.decision}" style="font-size:11px;margin-left:4px">${t.decision}</span></div>
      </div>
      <div class="panel-close" onclick="closePanel()">✕</div>
    </div>

    <div class="panel-section">
      <div class="panel-section-label">Anomaly score</div>
      <div class="gauge-wrap">
        <div class="risk-meter"><div class="risk-needle" style="left:${t.score * 100}%"></div></div>
        <div class="gauge-label-wrap"><span>0.0</span><span style="color:${sc};font-weight:600">${t.score.toFixed(3)}</span><span>1.0</span></div>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-section-label">Transaction data</div>
      <div class="panel-row"><span class="panel-key">ID</span><span class="panel-val">${t.id}</span></div>
      <div class="panel-row"><span class="panel-key">Amount</span><span class="panel-val">$${t.amount.toFixed(2)}</span></div>
      <div class="panel-row"><span class="panel-key">Hour</span><span class="panel-val">${String(t.hour).padStart(2, "0")}:00</span></div>
      <div class="panel-row"><span class="panel-key">Freq (1h / 24h)</span><span class="panel-val">${t.freq1h} / ${t.freq24h}</span></div>
      <div class="panel-row"><span class="panel-key">Country match</span><span class="panel-val" style="color:${t.cm ? "var(--green)" : "var(--red)"}">${t.cm ? "Home" : "Foreign"}</span></div>
      <div class="panel-row"><span class="panel-key">Merchant risk</span><span class="panel-val">${["Low", "Medium", "High"][t.mr]}</span></div>
      <div class="panel-row"><span class="panel-key">Prior declines</span><span class="panel-val" style="color:${t.dec > 2 ? "var(--red)" : "var(--text0)"}">${t.dec}</span></div>
    </div>

    <div class="panel-section">
      <div class="panel-section-label">Rules triggered (${t.rules.length})</div>
      <div class="rule-list">${ruleItems}</div>
    </div>

    <div class="panel-section">
      <div class="panel-section-label">Score vs recent transactions</div>
      <div class="chart-wrap"><canvas id="${ctx_id}" role="img" aria-label="Score comparison chart"></canvas></div>
    </div>

    <div class="action-row">
      <button class="btn btn-approve" onclick="takeAction('approve','${id}')">✓ Mark safe</button>
      <button class="btn btn-block"   onclick="takeAction('block','${id}')">✕ Block</button>
    </div>
  `;

  // Mini spark chart — this txn vs last 15
  const recent = allTxns.slice(0, 16);
  const labels = recent.map((x) => (x.id === t.id ? "▶ this" : x.id.slice(-4)));
  const data = recent.map((x) => x.score);
  const colors = recent.map((x) =>
    x.id === t.id ? "#4b9cf5" : scoreColor(x.score),
  );

  if (sparkChart) sparkChart.destroy();
  const ctx = document.getElementById(ctx_id);
  if (ctx) {
    sparkChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels.reverse(),
        datasets: [
          {
            data: data.reverse(),
            backgroundColor: colors.reverse(),
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (v) => " Score: " + v.raw.toFixed(3) },
          },
        },
        scales: {
          x: {
            ticks: {
              font: { family: "JetBrains Mono", size: 9 },
              color: "#5c6170",
              maxRotation: 45,
            },
            grid: { color: "rgba(255,255,255,0.04)" },
          },
          y: {
            min: 0,
            max: 1,
            ticks: {
              font: { family: "JetBrains Mono", size: 9 },
              color: "#5c6170",
            },
            grid: { color: "rgba(255,255,255,0.04)" },
          },
        },
      },
    });
  }
}

function closePanel() {
  selectedId = null;
  document.getElementById("panelEmpty").style.display = "flex";
  document.getElementById("panelBody").style.display = "none";
  if (sparkChart) {
    sparkChart.destroy();
    sparkChart = null;
  }
  renderTable();
}

function takeAction(action, id) {
  const t = allTxns.find((x) => x.id === id);
  if (!t) return;
  const msg =
    action === "approve"
      ? `${id} marked safe — removed from queue`
      : `${id} blocked — account flagged`;
  allTxns = allTxns.filter((x) => x.id !== id);
  closePanel();
  updateMetrics();
  renderTable();
  showToast(msg);
}

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2800);
}

// ── Boot ─────────────────────────────────────
// Pre-seed with 30 transactions
for (let i = 0; i < 30; i++) generateTxn(i * 7 + 13);
// Re-generate properly
txnCounter = 0;
allTxns = [];
for (let i = 0; i < 30; i++) allTxns.push(generateTxn(i * 7919 + 37));
allTxns.forEach((t) => (t.isNew = false));
updateMetrics();
renderTable();
startStream();
