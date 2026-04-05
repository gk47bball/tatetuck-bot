const state = {
  payload: null,
  selectedIdeaKey: null,
};

const summaryGrid = document.getElementById("summaryGrid");
const buySleeve = document.getElementById("buySleeve");
const sellSleeve = document.getElementById("sellSleeve");
const researchLongs = document.getElementById("researchLongs");
const researchShorts = document.getElementById("researchShorts");
const catalystRadar = document.getElementById("catalystRadar");
const tradeBlotter = document.getElementById("tradeBlotter");
const tradeBlotterNote = document.getElementById("tradeBlotterNote");
const validationPanel = document.getElementById("validationPanel");
const executionPanel = document.getElementById("executionPanel");
const systemBanner = document.getElementById("systemBanner");
const detailTitle = document.getElementById("detailTitle");
const detailSubtitle = document.getElementById("detailSubtitle");
const detailBody = document.getElementById("detailBody");
const freshnessPill = document.getElementById("freshnessPill");
const currentPlanNote = document.getElementById("currentPlanNote");
const researchNote = document.getElementById("researchNote");
const refreshButton = document.getElementById("refreshButton");

refreshButton.addEventListener("click", () => loadDashboard(true));
setInterval(() => loadDashboard(false), 60000);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: Math.abs(Number(value)) >= 1000 ? 0 : 2,
  }).format(Number(value));
}

function formatPct(value, digits = 1, signed = false) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const number = Number(value) * 100;
  const prefix = signed && number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(digits)}%`;
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

function formatBps(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${Number(value).toFixed(0)} bps`;
}

function formatDate(value) {
  const date = parseDashboardDate(value);
  if (!date) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatDay(value) {
  const date = parseDashboardDate(value);
  if (!date) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
}

function formatCompactDateTime(value) {
  const date = parseDashboardDate(value);
  if (!date) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatAge(days) {
  if (days === null || days === undefined || Number.isNaN(Number(days))) {
    return "freshness unknown";
  }
  if (Number(days) === 0) {
    return "today";
  }
  if (Number(days) === 1) {
    return "1 day old";
  }
  return `${Number(days)} days old`;
}

function badgeClass(kind) {
  if (kind === "long" || kind === "buy" || kind === "submitted") return "long";
  if (kind === "short" || kind === "sell") return "short";
  if (kind === "watch" || kind === "failed") return "warning";
  if (kind === "warning" || kind === "stale") return "warning";
  return "neutral";
}

function parseDashboardDate(value) {
  if (!value) {
    return null;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(value))) {
    const [year, month, day] = String(value).split("-").map(Number);
    return new Date(year, month - 1, day, 23, 59, 59);
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function referenceTimestamp() {
  return parseDashboardDate(state.payload?.summary?.generated_at) || new Date();
}

function humanizeToken(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeIdentityValue(value) {
  return String(value ?? "").trim().toLowerCase();
}

function renderIdentityStack(symbol, companyName, programName) {
  const lines = [
    { className: "idea-symbol", value: symbol },
    { className: "idea-company", value: companyName },
    { className: "idea-program", value: programName },
  ];
  const seen = new Set();
  return lines
    .filter(({ value }) => {
      const normalized = normalizeIdentityValue(value);
      if (!normalized || seen.has(normalized)) {
        return false;
      }
      seen.add(normalized);
      return true;
    })
    .map(({ className, value }) => `<div class="${className}">${escapeHtml(value)}</div>`)
    .join("");
}

function truncateText(value, maxLength = 120) {
  const text = String(value || "").trim();
  if (!text || text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

function eventTimingMeta(value) {
  const eventDate = parseDashboardDate(value);
  if (!eventDate) {
    return null;
  }
  const now = referenceTimestamp();
  const msPerDay = 24 * 60 * 60 * 1000;
  const deltaMs = eventDate.getTime() - now.getTime();
  const deltaDays = Math.ceil(deltaMs / msPerDay);
  if (deltaMs < 0) {
    return {
      label: "post-event",
      detail: `${Math.max(1, Math.ceil(Math.abs(deltaMs) / msPerDay))}d ago`,
      badgeClass: "warning",
      state: "post-event",
    };
  }
  if (deltaDays <= 3) {
    return {
      label: `T-${Math.max(0, deltaDays)}d`,
      detail: formatDay(value),
      badgeClass: "warning",
      state: "imminent",
    };
  }
  if (deltaDays <= 7) {
    return {
      label: `${deltaDays}d`,
      detail: formatDay(value),
      badgeClass: "neutral",
      state: "near-term",
    };
  }
  return {
    label: formatDay(value),
    detail: `${deltaDays}d out`,
    badgeClass: "neutral",
    state: "future",
  };
}

function compactMetric(label, value) {
  return `
    <div class="compact-metric">
      <div class="compact-label">${escapeHtml(label)}</div>
      <div class="compact-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function ideaSnapshotBadge(idea) {
  return idea.as_of ? `<span class="badge neutral">snapshot ${escapeHtml(formatDay(idea.as_of))}</span>` : "";
}

function ideaBadgeRow(idea, options = {}) {
  const timing = eventTimingMeta(idea.primary_event_date);
  const pills = [
    `<span class="badge neutral">${escapeHtml(idea.idea_level)}</span>`,
    idea.primary_event_type ? `<span class="badge neutral">${escapeHtml(idea.primary_event_type)}</span>` : "",
    !options.compact && idea.phase ? `<span class="badge neutral">${escapeHtml(idea.phase)}</span>` : "",
    !options.compact && idea.thesis_horizon ? `<span class="badge neutral">${escapeHtml(idea.thesis_horizon)}</span>` : "",
    idea.special_situation_label ? `<span class="badge warning">${escapeHtml(idea.special_situation_label)}</span>` : "",
    idea.deployable === false ? `<span class="badge warning">${escapeHtml(deploymentLabel(idea))}</span>` : "",
    idea.in_current_plan ? `<span class="badge long">in current plan</span>` : "",
    timing && timing.state === "post-event" ? `<span class="badge warning">${escapeHtml(timing.label)}</span>` : "",
    !options.compact && idea.evidence_count ? `<span class="badge neutral">${escapeHtml(`${idea.evidence_count} evidence`)}</span>` : "",
    ideaSnapshotBadge(idea),
  ];
  return pills.filter(Boolean).join("");
}

function ideaRankLabel(idea, index) {
  if (idea.direction === "long" && index < 3) {
    return `Top long #${index + 1}`;
  }
  if (idea.direction === "short" && index === 0) {
    return "Top short";
  }
  return "";
}

function eventToneClass(eventType) {
  const text = String(eventType || "").toLowerCase();
  if (text.includes("pdufa") || text.includes("approval") || text.includes("regulatory") || text.includes("label")) {
    return "regulatory";
  }
  if (text.includes("phase") || text.includes("readout") || text.includes("trial")) {
    return "clinical";
  }
  if (text.includes("commercial") || text.includes("earnings")) {
    return "commercial";
  }
  if (text.includes("transaction") || text.includes("adcom")) {
    return "transaction";
  }
  return "other";
}

function startOfWeek(date) {
  const output = new Date(date);
  const day = output.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  output.setDate(output.getDate() + diff);
  output.setHours(0, 0, 0, 0);
  return output;
}

function catalystWeekLabel(date) {
  const now = referenceTimestamp();
  const currentWeek = startOfWeek(now);
  const eventWeek = startOfWeek(date);
  const deltaWeeks = Math.round((eventWeek.getTime() - currentWeek.getTime()) / (7 * 24 * 60 * 60 * 1000));
  if (deltaWeeks === 0) {
    return "This week";
  }
  if (deltaWeeks === 1) {
    return "Next week";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(eventWeek);
}

function deploymentLabel(idea) {
  if (!idea || idea.deployable) {
    return "";
  }
  if (idea.deployment_status === "research_only_short") return "research-only short";
  if (idea.deployment_status === "research_only_program_short") return "program short";
  if (idea.deployment_status === "research_only_program") return "program research";
  if (idea.deployment_status === "research_artifact_program") return "study artifact";
  if (idea.deployment_status === "data_integrity_block") return "integrity block";
  if (idea.deployment_status === "watchlist_only") return "watchlist";
  return "research-only";
}

function metricBlock(label, value) {
  return `
    <div class="metric-block">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function emptyState(message) {
  return `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function combinedResearchIdeas() {
  if (!state.payload) {
    return [];
  }
  return [...(state.payload.research_book.company_ideas || []), ...(state.payload.research_book.program_ideas || [])];
}

function findTradeByKey(ideaKey) {
  return state.payload?.current_plan?.trade_rows?.find((row) => row.idea_key === ideaKey) ?? null;
}

function findResearchIdeaByKey(ideaKey) {
  return combinedResearchIdeas().find((idea) => idea.idea_key === ideaKey) ?? null;
}

function companyResearchIdeaForSymbol(symbol) {
  return (state.payload?.research_book?.company_ideas || []).find((idea) => idea.symbol === symbol) ?? null;
}

function latestTrade(symbol) {
  return state.payload?.current_plan?.trade_rows?.find((row) => row.symbol === symbol) ?? null;
}

function chooseDefaultIdeaKey(payload) {
  return (
    payload.current_plan.buy_orders[0]?.idea_key ||
    payload.current_plan.trade_rows[0]?.idea_key ||
    payload.research_book.longs[0]?.idea_key ||
    payload.research_book.shorts[0]?.idea_key ||
    null
  );
}

function renderSummary(payload) {
  const nav = payload.current_plan.nav;
  const cards = [
    {
      label: "Paper equity",
      value: nav ? formatMoney(nav.equity) : "—",
      subcopy: nav ? `${formatMoney(nav.cash)} cash` : "No portfolio snapshot available",
    },
    {
      label: "Gross target",
      value: formatPct(payload.summary.gross_target_weight / 100, 1),
      subcopy: formatMoney(payload.summary.gross_target_notional),
    },
    {
      label: "Live actions",
      value: String(payload.summary.buy_orders + payload.summary.sell_orders + payload.summary.hold_orders),
      subcopy: `${payload.summary.buy_orders} buys, ${payload.summary.sell_orders} exits`,
    },
    {
      label: "Research inventory",
      value: String(payload.summary.idea_count),
      subcopy: `${payload.summary.company_idea_count} company, ${payload.summary.program_idea_count} program`,
    },
    {
      label: "Validation rank IC",
      value: formatNumber(payload.summary.validation_rank_ic, 3),
      subcopy: `${formatPct(payload.summary.validation_exact_event_rate, 1)} exact event coverage`,
    },
    {
      label: "Research freshness",
      value: payload.summary.research_freshness_days === null ? "—" : `${payload.summary.research_freshness_days}d`,
      subcopy: payload.summary.research_notice || (payload.summary.research_as_of ? `as of ${formatDate(payload.summary.research_as_of)}` : "No research snapshot"),
    },
  ];

  summaryGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <div class="label">${escapeHtml(card.label)}</div>
          <div class="value">${escapeHtml(card.value)}</div>
          <div class="subcopy">${escapeHtml(card.subcopy)}</div>
        </article>
      `,
    )
    .join("");
}

function tradeCard(row) {
  const active = state.selectedIdeaKey === row.idea_key ? "active" : "";
  const expectedReturn = companyResearchIdeaForSymbol(row.symbol)?.expected_return ?? null;
  return `
    <article class="idea-card trade-card ${active}" data-idea-key="${escapeHtml(row.idea_key)}">
      <div class="idea-head">
        <div class="idea-company-stack">
          <div class="idea-symbol">${escapeHtml(row.symbol)}</div>
          <div class="idea-company">${escapeHtml(row.company_name || row.symbol)}</div>
        </div>
        <span class="badge ${badgeClass(row.stage)}">${escapeHtml(row.stage)}</span>
      </div>
      <div class="idea-metrics">
        ${metricBlock("Target weight", formatPct(row.scaled_target_weight / 100, 1))}
        ${metricBlock("Confidence", formatPct(row.confidence, 0))}
        ${metricBlock("Target notional", formatMoney(row.target_notional))}
        ${metricBlock("Expected return", formatPct(expectedReturn, 1, true))}
      </div>
      <p class="idea-summary">${escapeHtml(row.scenario || "No scenario attached")}</p>
      <div class="idea-foot">
        <div class="pill-row">
          <span class="badge ${badgeClass(row.action)}">${escapeHtml(row.action)}</span>
          ${row.execution_profile ? `<span class="badge neutral">${escapeHtml(row.execution_profile)}</span>` : ""}
          ${row.setup_type ? `<span class="badge neutral">${escapeHtml(row.setup_type)}</span>` : ""}
        </div>
      </div>
    </article>
  `;
}

function researchMetricBlocks(idea) {
  if (idea.idea_level === "program") {
    return `
      ${metricBlock("Expected return", formatPct(idea.expected_return, 1, true))}
      ${metricBlock("Confidence", formatPct(idea.confidence, 0))}
      ${metricBlock("Catalyst PoS", formatPct(idea.catalyst_success_prob, 0))}
      ${metricBlock("Event date", formatDate(idea.primary_event_date))}
    `;
  }
  return `
    ${metricBlock("Expected return", formatPct(idea.expected_return, 1, true))}
    ${metricBlock("Confidence", formatPct(idea.confidence, 0))}
    ${metricBlock("Target weight", formatPct(idea.target_weight / 100, 1))}
    ${metricBlock("Catalyst PoS", formatPct(idea.catalyst_success_prob, 0))}
  `;
}

function featuredResearchCard(idea, index) {
  const active = state.selectedIdeaKey === idea.idea_key ? "active" : "";
  const cardClass = idea.direction === "short" ? "short-card" : "long-card";
  const timing = eventTimingMeta(idea.primary_event_date);
  const rankLabel = ideaRankLabel(idea, index);
  return `
    <article class="idea-card featured-card ${cardClass} ${active} ${timing?.state === "post-event" ? "post-event-card" : ""}" data-idea-key="${escapeHtml(idea.idea_key)}">
      <div class="idea-head">
        <div class="idea-company-stack">
          ${renderIdentityStack(idea.symbol, idea.company_name, idea.idea_level === "program" ? idea.program_name : null)}
        </div>
        <div class="card-badge-stack">
          <span class="badge ${badgeClass(idea.direction)}">${escapeHtml(idea.direction)}</span>
          ${rankLabel ? `<span class="badge neutral">${escapeHtml(rankLabel)}</span>` : ""}
        </div>
      </div>
      <div class="idea-metrics">
        ${researchMetricBlocks(idea)}
      </div>
      <p class="idea-summary">${escapeHtml(idea.rationale_preview || idea.scenario || "No thesis summary yet")}</p>
      <div class="idea-foot">
        <div class="pill-row">
          ${ideaBadgeRow(idea)}
          ${timing && timing.state !== "post-event" ? `<span class="badge ${timing.badgeClass}">${escapeHtml(timing.label)}</span>` : ""}
        </div>
      </div>
    </article>
  `;
}

function compactResearchCard(idea) {
  const active = state.selectedIdeaKey === idea.idea_key ? "active" : "";
  const cardClass = idea.direction === "short" ? "short-card" : "long-card";
  const timing = eventTimingMeta(idea.primary_event_date);
  return `
    <article class="idea-card compact-card ${cardClass} ${active} ${timing?.state === "post-event" ? "post-event-card" : ""}" data-idea-key="${escapeHtml(idea.idea_key)}">
      <div class="compact-card-top">
        <div class="idea-company-stack">
          ${renderIdentityStack(idea.symbol, idea.company_name, idea.idea_level === "program" ? idea.program_name : null)}
        </div>
        <div class="card-badge-stack">
          <span class="badge ${badgeClass(idea.direction)}">${escapeHtml(idea.direction)}</span>
          ${timing ? `<span class="badge ${timing.badgeClass}">${escapeHtml(timing.label)}</span>` : ""}
        </div>
      </div>
      <div class="compact-metric-strip">
        ${compactMetric("Expected", formatPct(idea.expected_return, 1, true))}
        ${compactMetric("Conf", formatPct(idea.confidence, 0))}
        ${compactMetric("Event", idea.primary_event_date ? formatDay(idea.primary_event_date) : "—")}
      </div>
      <div class="compact-summary">${escapeHtml(truncateText(idea.rationale_preview || idea.scenario || "No thesis summary yet", 110))}</div>
      <div class="pill-row">
        ${ideaBadgeRow(idea, { compact: true })}
      </div>
    </article>
  `;
}

function renderResearchDeck(ideas, options = {}) {
  if (!ideas.length) {
    return emptyState(options.emptyMessage || "No ideas are available.");
  }
  const featuredCount = Math.min(options.featuredCount || 0, ideas.length);
  const featured = ideas.slice(0, featuredCount);
  const compact = ideas.slice(featuredCount);
  return `
    ${featured.length ? `<div class="featured-idea-grid">${featured.map((idea, index) => featuredResearchCard(idea, index)).join("")}</div>` : ""}
    ${compact.length ? `<div class="compact-idea-list">${compact.map(compactResearchCard).join("")}</div>` : ""}
  `;
}

function renderCurrentPlan(payload) {
  const reconciliation = payload.current_plan.reconciliation;
  const parts = [];
  if (payload.current_plan.run) {
    parts.push(`${payload.current_plan.run.status} run`);
    if (payload.current_plan.run.metrics?.actionable_instructions !== undefined) {
      parts.push(`${payload.current_plan.run.metrics.actionable_instructions} actionable`);
    }
    parts.push(formatAge(payload.current_plan.freshness_days));
    if (payload.current_plan.run.status === "failed") {
      parts.push(payload.current_plan.run.notes?.includes("ConnectionError") ? "broker connection failed" : "run failed");
    }
  } else {
    parts.push("No trade run metadata found");
  }
  if (reconciliation) {
    parts.push(`${reconciliation.blocker_count} reconciliation blockers`);
    if ((reconciliation.missing_symbols || []).length) {
      parts.push(`${reconciliation.missing_symbols.length} missing symbols`);
    }
  }
  currentPlanNote.textContent = parts.join(" · ");

  buySleeve.innerHTML = payload.current_plan.buy_orders.length
    ? payload.current_plan.buy_orders.map(tradeCard).join("")
    : emptyState(
        payload.current_plan.run?.status === "failed"
          ? "The latest trade run failed before it produced a live buy sleeve."
          : "No active buy orders in the latest PM trade run.",
      );

  const exits = [...payload.current_plan.sell_orders, ...payload.current_plan.hold_orders];
  sellSleeve.innerHTML = exits.length
    ? exits.map(tradeCard).join("")
    : emptyState(
        payload.current_plan.run?.status === "failed"
          ? "No exit or hold instructions were produced in the failed run."
          : "No exits or holdovers in the latest PM trade run.",
      );
}

function renderResearch(payload) {
  const noteParts = [];
  if (payload.research_book.as_of) {
    noteParts.push(`latest research snapshot ${formatDate(payload.research_book.as_of)}`);
    noteParts.push(formatAge(payload.research_book.freshness_days));
  }
  if (payload.research_book.notice) {
    noteParts.push(payload.research_book.notice);
  }
  researchNote.textContent = noteParts.join(" · ") || "No research snapshot found";

  researchLongs.innerHTML = renderResearchDeck(payload.research_book.longs, {
    featuredCount: 3,
    emptyMessage: "No long ideas are available in the current research deck.",
  });
  researchShorts.innerHTML = renderResearchDeck(payload.research_book.shorts, {
    featuredCount: 1,
    emptyMessage: "No short sleeve ideas are currently available in the research deck.",
  });
}

function renderSystemBanner(payload) {
  const rawDecision = payload.validation?.promotion?.decision || "unpromoted";
  const decision = humanizeToken(rawDecision);
  const blockers = payload.validation?.promotion?.blockers || [];
  const promote = rawDecision === "promote";
  const paperReady = rawDecision === "paper_trade_ready";
  const positiveDecision = promote || paperReady;
  const rationale = payload.validation?.promotion?.rationale
    || (promote
      ? "Latest validation clears the promotion gate."
      : paperReady
        ? "Latest validation clears the paper-trading gate, but not the A-grade capital gate."
        : "Latest validation does not clear the promotion gate.");
  const bannerClass = positiveDecision ? "system-banner-ok" : "system-banner-warning";
  const detailParts = [];
  if (payload.validation.as_of) {
    detailParts.push(`validation ${formatAge(payload.validation.freshness_days)}`);
  }
  if (blockers.length) {
    detailParts.push(`${blockers.length} blocker${blockers.length === 1 ? "" : "s"}`);
  }
  if (payload.current_plan.run?.status === "failed") {
    detailParts.push("latest trade run failed");
  }
  systemBanner.innerHTML = `
    <div class="system-banner ${bannerClass}">
      <div class="system-banner-head">
        <div>
          <div class="system-banner-kicker">Model gate</div>
          <div class="system-banner-title">${escapeHtml(decision)}</div>
        </div>
        <div class="pill-row">
          <span class="badge ${positiveDecision ? "long" : "warning"}">${escapeHtml(decision)}</span>
          ${payload.validation.as_of ? `<span class="badge neutral">${escapeHtml(formatAge(payload.validation.freshness_days))}</span>` : ""}
          ${blockers.length ? `<span class="badge warning">${escapeHtml(`${blockers.length} blockers`)}</span>` : ""}
        </div>
      </div>
      <div class="system-banner-copy">${escapeHtml(rationale)}</div>
      <div class="system-banner-meta">
        ${escapeHtml(detailParts.join(" · ") || "No validation status was persisted.")}
      </div>
    </div>
  `;
}

function renderCatalystRadar(payload) {
  if (!payload.research_book.catalyst_calendar.length) {
    catalystRadar.innerHTML = emptyState("No dated catalysts are available in the latest research snapshot.");
    return;
  }

  const grouped = new Map();
  payload.research_book.catalyst_calendar.forEach((event) => {
    const eventDate = parseDashboardDate(event.event_date);
    if (!eventDate) {
      return;
    }
    const label = catalystWeekLabel(eventDate);
    if (!grouped.has(label)) {
      grouped.set(label, []);
    }
    grouped.get(label).push(event);
  });

  catalystRadar.innerHTML = [...grouped.entries()]
    .map(
      ([label, events]) => `
        <section class="timeline-week">
          <div class="timeline-week-label">${escapeHtml(label)}</div>
          <div class="timeline-week-list">
            ${events
              .map((event) => {
                const timing = eventTimingMeta(event.event_date);
                return `
                  <article class="timeline-row" data-idea-key="${escapeHtml(event.idea_key || "")}">
                    <div class="event-rail ${escapeHtml(eventToneClass(event.event_type))}"></div>
                    <div class="timeline-row-body">
                      <div class="timeline-row-head">
                        <div>
                          <div class="timeline-symbol">${escapeHtml(event.symbol)}</div>
                          <div class="timeline-title">${escapeHtml(event.program_name || humanizeToken(event.event_type) || "unspecified catalyst")}</div>
                        </div>
                        <div class="timeline-date-stack">
                          <div class="timeline-date">${escapeHtml(formatDay(event.event_date))}</div>
                          ${timing?.detail ? `<div class="timeline-countdown">${escapeHtml(timing.detail)}</div>` : ""}
                        </div>
                      </div>
                      <div class="timeline-meta">
                        ${escapeHtml(event.company_name || event.symbol)} · ${escapeHtml(humanizeToken(event.event_type) || "unspecified catalyst")}
                      </div>
                      <div class="pill-row">
                        <span class="badge ${event.event_exact ? "long" : "warning"}">${event.event_exact ? "exact" : "estimated"}</span>
                        <span class="badge ${event.in_current_plan ? "long" : "neutral"}">${escapeHtml(event.in_current_plan ? "book" : "watchlist")}</span>
                        ${event.direction ? `<span class="badge ${badgeClass(event.direction)}">${escapeHtml(event.direction)}</span>` : ""}
                      </div>
                    </div>
                  </article>
                `;
              })
              .join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

function blotterRow(row) {
  const pnl = row.pnl?.mark_to_market_net_return;
  const costs = [
    row.expected_slippage_bps !== null ? formatBps(row.expected_slippage_bps) : null,
    row.expected_round_trip_cost_bps !== null ? formatBps(row.expected_round_trip_cost_bps) : null,
  ]
    .filter(Boolean)
    .join(" / ");

  return `
    <tr>
      <td class="time-cell">${escapeHtml(formatCompactDateTime(row.planned_at))}</td>
      <td><span class="idea-symbol">${escapeHtml(row.symbol)}</span></td>
      <td><span class="badge ${badgeClass(row.action)}">${escapeHtml(row.action)}</span></td>
      <td>${escapeHtml(row.scenario || "—")}</td>
      <td>${escapeHtml(formatPct(row.scaled_target_weight / 100, 1))}</td>
      <td><span class="badge ${badgeClass(row.stage)}">${escapeHtml(row.stage)}</span></td>
      <td>${escapeHtml(costs || "—")}</td>
      <td>${escapeHtml(pnl === null || pnl === undefined ? "—" : formatPct(pnl, 1, true))}</td>
    </tr>
  `;
}

function renderTradeBlotter(payload) {
  const rows = payload.current_plan.trade_rows.length
    ? payload.current_plan.trade_rows
    : (payload.current_plan.recent_trade_rows || []);
  tradeBlotterNote.textContent = payload.current_plan.trade_rows_source === "current_run"
    ? "Showing the latest run instructions."
    : payload.current_plan.trade_rows_source === "latest_activity"
      ? "Current run had no matched orders. Showing the latest recorded trade activity."
      : "No recorded trade activity.";

  tradeBlotter.innerHTML = rows.length
    ? rows.map(blotterRow).join("")
    : `<tr><td colspan="8">${escapeHtml("No trade instructions are available.")}</td></tr>`;
}

function renderValidation(payload) {
  const rawDecision = payload.validation.promotion?.decision || "unpromoted";
  const positiveDecision = rawDecision === "promote" || rawDecision === "paper_trade_ready";
  const blockers = payload.validation.promotion?.blockers ?? [];
  validationPanel.innerHTML = `
    <div class="validation-card">
      <div class="validation-alert ${positiveDecision ? "ok" : "warning"}">
        <div class="validation-alert-title">${escapeHtml(humanizeToken(rawDecision))}</div>
        <div class="validation-alert-copy">${escapeHtml(payload.validation.promotion?.rationale || "No promotion note persisted.")}</div>
      </div>
      <div class="validation-grid">
        <div class="small-metric">
          <div class="metric-label">Rank IC</div>
          <div class="metric-value">${escapeHtml(formatNumber(payload.validation.rank_ic, 3))}</div>
        </div>
        <div class="small-metric">
          <div class="metric-label">Cost-adjusted spread</div>
          <div class="metric-value">${escapeHtml(formatPct(payload.validation.cost_adjusted_top_bottom_spread, 1, true))}</div>
        </div>
        <div class="small-metric">
          <div class="metric-label">Exact event rate</div>
          <div class="metric-value">${escapeHtml(formatPct(payload.validation.exact_primary_event_rate, 1))}</div>
        </div>
        <div class="small-metric">
          <div class="metric-label">Windows</div>
          <div class="metric-value">${escapeHtml(String(payload.validation.windows || 0))}</div>
        </div>
      </div>
      <div class="pill-row">
        <span class="badge ${positiveDecision ? "long" : "warning"}">
          ${escapeHtml(humanizeToken(rawDecision))}
        </span>
        ${payload.validation.as_of ? `<span class="badge neutral">${escapeHtml(formatAge(payload.validation.freshness_days))}</span>` : ""}
      </div>
      <div class="detail-section">
        <div class="metric-label">Blockers</div>
        <div class="detail-list">
          ${
            blockers.length
              ? blockers.slice(0, 5).map((blocker) => `<div>${escapeHtml(blocker)}</div>`).join("")
              : `<div>${escapeHtml("No blockers were persisted on the latest promotion row.")}</div>`
          }
        </div>
      </div>
    </div>
  `;
}

function renderExecution(payload) {
  if (!payload.execution.scorecards.length) {
    executionPanel.innerHTML = emptyState(
      payload.execution.feedback_rows
        ? "Execution feedback exists, but no profile scorecards have matured yet."
        : "No realized execution feedback is available yet. The dashboard will surface IR, hit rate, and cost-aware returns once the book has true trade-date history.",
    );
    return;
  }

  executionPanel.innerHTML = `
    <div class="scorecard-list">
      ${payload.execution.scorecards
        .map(
          (row) => `
            <div class="scorecard-row">
              <div class="idea-head">
                <div>
                  <div class="timeline-title">${escapeHtml(row.execution_profile)}</div>
                  <div class="timeline-meta">${escapeHtml(`${row.trades} trades`)}</div>
                </div>
                <span class="badge neutral">${escapeHtml(formatBps(row.avg_estimated_round_trip_cost_bps))}</span>
              </div>
              <div class="validation-grid">
                <div class="small-metric">
                  <div class="metric-label">MTM net</div>
                  <div class="metric-value">${escapeHtml(formatPct(row.avg_mark_to_market_net_return, 1, true))}</div>
                </div>
                <div class="small-metric">
                  <div class="metric-label">30d net</div>
                  <div class="metric-value">${escapeHtml(formatPct(row.avg_net_return_30d, 1, true))}</div>
                </div>
              </div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function buildAutoRationale(context) {
  const items = [];
  if (context.program_name) {
    items.push(`Program focus: ${context.program_name}`);
  }
  if (context.indication) {
    items.push(`Primary indication: ${context.indication}`);
  }
  if (context.primary_event_type) {
    items.push(`Catalyst: ${context.primary_event_type}`);
  }
  if (context.primary_event_date) {
    items.push(`Expected timing: ${formatDate(context.primary_event_date)}`);
  }
  if (context.phase) {
    items.push(`Development stage: ${context.phase}`);
  }
  if (context.scenario) {
    items.push(`Scenario: ${context.scenario}`);
  }
  return items;
}

function renderEvidenceItems(items) {
  if (!items || !items.length) {
    return `<div>${escapeHtml("No supporting evidence is currently attached to this idea.")}</div>`;
  }
  return items
    .slice(0, 3)
    .map((item) => {
      const title = item.url
        ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title || item.source || "source")}</a>`
        : escapeHtml(item.title || item.source || "source");
      const meta = [item.source, item.as_of].filter(Boolean).join(" · ");
      return `
        <div class="evidence-item">
          <div class="evidence-title">${title}</div>
          ${meta ? `<div class="timeline-meta">${escapeHtml(meta)}</div>` : ""}
          ${item.excerpt ? `<div class="detail-copy">${escapeHtml(item.excerpt)}</div>` : ""}
        </div>
      `;
    })
    .join("");
}

function renderDetail(selectedIdeaKey) {
  const selectedTrade = findTradeByKey(selectedIdeaKey);
  const selectedIdea = findResearchIdeaByKey(selectedIdeaKey);

  let context = null;
  if (selectedTrade) {
    const research = companyResearchIdeaForSymbol(selectedTrade.symbol) ?? {};
    context = {
      ...research,
      ...selectedTrade,
      current_trade: selectedTrade,
      symbol: selectedTrade.symbol,
      company_name: selectedTrade.company_name || research.company_name || selectedTrade.symbol,
    };
  } else if (selectedIdea) {
    context = {
      ...selectedIdea,
      current_trade: latestTrade(selectedIdea.symbol),
    };
  }

  if (!context || !context.symbol) {
    detailTitle.textContent = "No idea selected";
    detailSubtitle.textContent = "Choose a card to inspect the thesis, catalyst, and risk frame.";
    detailBody.innerHTML = "";
    return;
  }

  const titleBits = [context.symbol];
  if (context.program_name) {
    titleBits.push(context.program_name);
  } else if (context.company_name) {
    titleBits.push(context.company_name);
  }
  detailTitle.textContent = titleBits.join(" · ");
  detailSubtitle.textContent = context.program_name ? (context.company_name || context.symbol) : (context.scenario || "No scenario has been attached yet.");

  const rationaleItems = (context.rationale && context.rationale.length ? context.rationale : buildAutoRationale(context))
    .slice(0, 6)
    .map((item) => `<div>${escapeHtml(item)}</div>`)
    .join("");

  const riskFlags = [...new Set([...(context.risk_flags || [])])];
  const eventTiming = eventTimingMeta(context.primary_event_date);
  if ((context.crowding_risk ?? 0) >= 0.6) {
    riskFlags.push(`Crowding risk elevated (${formatPct(context.crowding_risk, 0)})`);
  }
  if ((context.financing_risk ?? 0) >= 0.6) {
    riskFlags.push(`Financing risk elevated (${formatPct(context.financing_risk, 0)})`);
  }
  if (eventTiming?.state === "post-event") {
    riskFlags.push("Primary catalyst date has passed; refresh the thesis before sizing.");
  }
  const evidenceSources = (context.evidence_sources || []).join(", ");

  const trade = context.current_trade ?? {};

  detailBody.innerHTML = `
    <section class="detail-section">
      <div class="pill-row">
        ${context.direction ? `<span class="badge ${badgeClass(context.direction)}">${escapeHtml(context.direction)}</span>` : ""}
        ${context.idea_level ? `<span class="badge neutral">${escapeHtml(context.idea_level)}</span>` : ""}
        ${context.deployable === false ? `<span class="badge warning">${escapeHtml(deploymentLabel(context))}</span>` : ""}
        ${trade.stage ? `<span class="badge ${badgeClass(trade.stage)}">${escapeHtml(trade.stage)}</span>` : ""}
        ${context.primary_event_type ? `<span class="badge neutral">${escapeHtml(context.primary_event_type)}</span>` : ""}
        ${context.primary_event_exact ? `<span class="badge long">exact catalyst</span>` : ""}
        ${eventTiming?.state === "post-event" ? `<span class="badge warning">${escapeHtml(eventTiming.label)}</span>` : ""}
        ${context.special_situation_label ? `<span class="badge warning">${escapeHtml(context.special_situation_label)}</span>` : ""}
      </div>
    </section>

    <section class="detail-section">
      <div class="detail-grid">
        ${metricBlock("Expected return", formatPct(context.expected_return, 1, true))}
        ${metricBlock("Confidence", formatPct(context.confidence ?? trade.confidence, 0))}
        ${metricBlock("Catalyst PoS", formatPct(context.catalyst_success_prob, 0))}
        ${metricBlock("Target weight", formatPct(((trade.scaled_target_weight ?? context.target_weight) || 0) / 100, 1))}
        ${metricBlock("Primary event date", formatDate(context.primary_event_date))}
        ${metricBlock("Thesis horizon", context.thesis_horizon || "—")}
        ${metricBlock("Research as of", formatDay(context.as_of))}
        ${metricBlock("Market cap", formatMoney(context.market_cap))}
        ${metricBlock("Company state", context.company_state || "—")}
        ${metricBlock("Price now", formatMoney(context.price_now))}
      </div>
    </section>

    <section class="detail-section">
      <div class="metric-label">Why it is in view</div>
      <div class="detail-list">
        ${rationaleItems || `<div>${escapeHtml("No rationale was stored for this idea.")}</div>`}
      </div>
    </section>

    <section class="detail-section">
      <div class="metric-label">Scientific frame</div>
      <div class="detail-list">
        <div>${escapeHtml(`Lead trial: ${context.lead_trial_title || "not attached"}`)}</div>
        <div>${escapeHtml(`Trial phase/status: ${context.lead_trial_phase || context.phase || "—"} / ${context.lead_trial_status || "—"}`)}</div>
        <div>${escapeHtml(`Primary outcomes: ${(context.lead_trial_primary_outcomes || []).join("; ") || "not attached"}`)}</div>
        <div>${escapeHtml(`Conditions: ${(context.program_conditions || []).join(", ") || context.indication || "not attached"}`)}</div>
        <div>${escapeHtml(`Special situation: ${context.special_situation_reason || context.special_situation_label || "none attached"}`)}</div>
        <div>${escapeHtml(`PM status: ${context.deployment_note || "No deployment note attached."}`)}</div>
      </div>
    </section>

    <section class="detail-section">
      <div class="metric-label">Evidence</div>
      <div class="detail-list">
        <div>${escapeHtml(`Evidence sources: ${evidenceSources || "none attached"}`)}</div>
        <div>${escapeHtml(`Evidence count: ${String(context.evidence_count || 0)}`)}</div>
      </div>
      <div class="detail-list">
        ${renderEvidenceItems(context.evidence_items || [])}
      </div>
    </section>

    <section class="detail-section">
      <div class="metric-label">Risk frame</div>
      <div class="risk-grid">
        ${
          riskFlags.length
            ? riskFlags.slice(0, 6).map((flag) => `<span class="risk-chip warning">${escapeHtml(flag)}</span>`).join("")
            : `<span class="risk-chip neutral">No explicit risk flags persisted</span>`
        }
      </div>
    </section>

    <section class="detail-section">
      <div class="metric-label">Trade context</div>
      <div class="detail-list">
        <div>${escapeHtml(`Trade action: ${trade.action || "not in current plan"}`)}</div>
        <div>${escapeHtml(`Target notional: ${formatMoney(trade.target_notional)}`)}</div>
        <div>${escapeHtml(`Expected slippage: ${formatBps(trade.expected_slippage_bps)}`)}</div>
        <div>${escapeHtml(`Round-trip cost: ${formatBps(trade.expected_round_trip_cost_bps)}`)}</div>
      </div>
    </section>
  `;
}

function wireCardClicks() {
  document.querySelectorAll("[data-idea-key]").forEach((node) => {
    node.addEventListener("click", () => {
      state.selectedIdeaKey = node.dataset.ideaKey;
      renderAll();
    });
  });
}

function renderAll() {
  if (!state.payload) {
    return;
  }
  renderSystemBanner(state.payload);
  renderSummary(state.payload);
  renderCurrentPlan(state.payload);
  renderResearch(state.payload);
  renderCatalystRadar(state.payload);
  renderTradeBlotter(state.payload);
  renderValidation(state.payload);
  renderExecution(state.payload);
  renderDetail(state.selectedIdeaKey);
  freshnessPill.textContent = state.payload.current_plan.run
    ? `${state.payload.current_plan.run.status} trade run · research ${formatAge(state.payload.research_book.freshness_days)}`
    : `research ${formatAge(state.payload.research_book.freshness_days)}`;
  wireCardClicks();
}

async function loadDashboard(showBusy) {
  if (showBusy) {
    freshnessPill.textContent = "Refreshing dashboard…";
  }
  try {
    const response = await fetch("/api/dashboard", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.payload = await response.json();
    if (!state.selectedIdeaKey || (!findTradeByKey(state.selectedIdeaKey) && !findResearchIdeaByKey(state.selectedIdeaKey))) {
      state.selectedIdeaKey = chooseDefaultIdeaKey(state.payload);
    }
    renderAll();
  } catch (error) {
    freshnessPill.textContent = `Dashboard load failed: ${error.message}`;
  }
}

loadDashboard(true);
