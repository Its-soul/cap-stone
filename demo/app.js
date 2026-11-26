"use strict";

const claimForm = document.querySelector("#claim-form");
const claimFields = document.querySelector("#claim-fields");
const claimAction = document.querySelector("#claim-action");
const newClaim = document.querySelector("#new-claim");
const claimMessage = document.querySelector("#claim-message");
const currency = new Intl.NumberFormat("en-IN", {style: "currency", currency: "INR", maximumFractionDigits: 0});
const wholeNumber = new Intl.NumberFormat("en-IN");
let groups = [];
let claimId = null;

function element(tag, content, className) {
    const node = document.createElement(tag);
    if (content !== undefined) node.textContent = content;
    if (className) node.className = className;
    return node;
}

function title(value) {
    return value[0].toUpperCase() + value.slice(1);
}

function formatFact(value) {
    return typeof value === "boolean" ? (value ? "Yes" : "No") : currency.format(value);
}

function message(node, value, isError = false) {
    node.textContent = value;
    node.hidden = !value;
    node.classList.toggle("error", isError);
}

async function request(path, data) {
    const options = data === undefined ? {} : {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data),
    };
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    return result;
}

function buildField(field) {
    const label = element("label", field.label);
    if (typeof field.default === "boolean") {
        const select = element("select");
        select.name = field.key;
        for (const value of [true, false]) {
            const option = element("option", formatFact(value));
            option.value = String(value);
            option.defaultSelected = value === field.default;
            select.append(option);
        }
        label.append(select);
    } else {
        const wrapper = element("span", undefined, "money-input");
        const symbol = element("span", "₹");
        symbol.setAttribute("aria-hidden", "true");
        const input = element("input");
        input.name = field.key;
        input.inputMode = "numeric";
        input.required = true;
        input.defaultValue = wholeNumber.format(field.default);
        input.addEventListener("blur", () => {
            const value = parseAmount(input.value);
            if (value !== null) input.value = wholeNumber.format(value);
        });
        wrapper.append(symbol, input);
        label.append(wrapper);
    }
    return label;
}

function parseAmount(text) {
    const digits = text.trim().replaceAll(",", "");
    const value = Number(digits);
    return /^\d+$/.test(digits) && Number.isSafeInteger(value) && value > 0 ? value : null;
}

function buildForm() {
    for (const group of groups) {
        const section = element("div", undefined, "field-group");
        const heading = element("h3", title(group.role));
        heading.id = `fields-${group.role}`;
        section.setAttribute("role", "group");
        section.setAttribute("aria-labelledby", heading.id);
        const grid = element("div", undefined, "field-grid");
        grid.append(...group.fields.map(buildField));
        section.append(heading, grid);
        claimFields.append(section);
    }
}

function readFacts() {
    const formData = new FormData(claimForm);
    const facts = {};
    for (const {fields} of groups) {
        for (const field of fields) {
            const raw = formData.get(field.key);
            const value = typeof field.default === "boolean" ? raw === "true" : parseAmount(raw);
            if (value === null) {
                claimForm.elements.namedItem(field.key).focus();
                throw new Error(`${field.label} must be a positive whole number in rupees.`);
            }
            facts[field.key] = value;
        }
    }
    return facts;
}

function renderDecision(id, heading, data, emptyText) {
    const card = document.querySelector(id);
    card.replaceChildren(element("h3", heading));
    if (data) {
        card.append(
            element("span", data.decision, `pill ${data.decision.toLowerCase()}`),
            element("p", formatFact(data.amount_inr), "amount"),
            element("p", data.reason, "reason"),
        );
    } else {
        card.append(element("p", emptyText, "muted"));
    }
}

function renderAudit(result) {
    const container = document.querySelector("#audit-content");
    if (!result) {
        container.replaceChildren(element("p", "No claim has been run."));
        return;
    }
    const fields = Object.fromEntries(groups.flatMap(group => group.fields.map(field => [field.key, field.label])));
    const evidence = element("ul");
    for (const check of result.checks) {
        const item = element("li");
        const facts = Object.entries(check.evidence).map(([key, value]) => `${fields[key]} ${formatFact(value)}`);
        const binding = `assessment v${check.assessment.version}, source v${check.input.version}`;
        item.append(element("strong", `${title(check.role)} · ${title(check.finding.toLowerCase())}`),
            ` (${binding}${check.valid ? "" : ", invalid"})`,
            ...facts.map((fact, index) => `${index === 0 ? " — " : ", "}${fact}`));
        evidence.append(item);
    }
    const eventCount = result.audit.events.length + result.audit.engine_events.length;
    const events = element("details");
    events.append(element("summary", `View event data (${eventCount})`),
        element("pre", JSON.stringify(result.audit, null, 2)));
    container.replaceChildren(element("h4", "Evidence bindings"), evidence,
        eventCount ? events : element("p", "No new audit events."));
}

function certificateLabel(certificate) {
    if (!certificate) return "—";
    return `v${certificate.version}${certificate.status === "VALID" ? "" : ` (${certificate.status})`}`;
}

function renderClaim(result) {
    renderDecision("#previous-decision", result?.previous_historical ? "Previous · Historical" : "Previous",
        result?.previous, "No previous decision");
    renderDecision("#current-decision", "Current", result?.current,
        result ? "No current decision" : "Run a claim to see its decision.");

    const recovery = result?.recovery;
    const blocked = recovery && recovery.status !== "COMPLETE";
    message(document.querySelector("#recovery-status"), blocked
        ? `Recovery ${recovery.status}. ${recovery.blocked.map(([artifact, reason]) => `${artifact} · ${reason}`).join(". ")}`
        : "", Boolean(blocked));

    const changes = document.querySelector("#changes");
    if (result?.changes.length) {
        changes.replaceChildren(...result.changes.map(change => {
            const row = element("div", undefined, "change");
            row.append(element("span", change.label), element("span",
                `${formatFact(change.before)} → ${formatFact(change.after)}`, "change-values"));
            return row;
        }));
    } else {
        changes.replaceChildren(element("p", result?.no_changes ? "No changes detected" : "No edits yet", "empty"));
    }

    document.querySelector("#checks-heading").textContent = result?.previous ? "Checks after the edit" : "Specialist checks";
    const checks = result?.checks ?? groups.map(group => ({role: group.role, status: "—"}));
    document.querySelector("#check-rows").replaceChildren(...checks.map(check => {
        const row = element("tr");
        const status = element("td");
        status.append(element("span", check.status, result ? `pill ${check.status.toLowerCase()}` : "muted"));
        row.append(element("td", title(check.role)), status);
        return row;
    }));

    const certificate = result?.certificate;
    document.querySelector("#certificate-version").textContent = certificate?.before
        ? `${certificateLabel(certificate.before)} → ${certificateLabel(certificate.after)}`
        : certificateLabel(certificate?.after);
    renderAudit(result);
}

function setClaimBusy(busy) {
    claimFields.inert = busy;
    claimAction.disabled = busy;
    newClaim.disabled = busy;
    claimAction.textContent = busy ? "Running…" : claimId ? "Apply changes" : "Run claim";
}

claimForm.addEventListener("submit", async event => {
    event.preventDefault();
    message(claimMessage, "");
    try {
        const facts = readFacts();
        setClaimBusy(true);
        const result = await request(claimId ? `/api/claims/${claimId}` : "/api/claims", facts);
        claimId = result.claim_id;
        renderClaim(result);
    } catch (error) {
        message(claimMessage, error.message, true);
    } finally {
        setClaimBusy(false);
    }
});

newClaim.addEventListener("click", () => {
    claimId = null;
    claimForm.reset();
    document.querySelector("#audit").open = false;
    message(claimMessage, "");
    renderClaim(null);
    setClaimBusy(false);
    claimForm.elements.namedItem("amount_inr").focus();
});

const v2Form = document.querySelector("#v2-form");
const v2Action = document.querySelector("#v2-action");
const v2Result = document.querySelector("#v2-result");

v2Form.addEventListener("input", () => message(v2Result, ""));
v2Form.addEventListener("submit", async event => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(v2Form));
    const inputs = v2Form.querySelectorAll("textarea");
    inputs.forEach(input => { input.readOnly = true; });
    v2Action.disabled = true;
    message(v2Result, "Running V2…");
    try {
        const result = await request("/api/v2", data);
        if (result.status === "unavailable") {
            message(v2Result, `V2 unavailable. ${result.reason}`, true);
        } else {
            const scores = element("dl", undefined, "scores");
            for (const label of ["depends_on", "independent"]) {
                const row = element("div");
                const score = element("dd", String(result.scores[label]));
                score.dataset.score = label;
                row.append(element("dt", label), score);
                scores.append(row);
            }
            v2Result.replaceChildren(element("strong", result.prediction), scores,
                element("p", `Saved threshold ${result.threshold}`, "muted"));
        }
    } catch (error) {
        message(v2Result, error.message, true);
    } finally {
        v2Action.disabled = false;
        inputs.forEach(input => { input.readOnly = false; });
    }
});

async function initialize() {
    if (window.location.protocol === "file:") {
        message(
            claimMessage,
            'Open this demo through its local Python server: run "python scripts/run_claim_demo.py" from the repository root, then open http://127.0.0.1:8000/.',
            true
        );
        return;
    }
    try {
        groups = await request("/api/form");
        buildForm();
        renderClaim(null);
        setClaimBusy(false);
    } catch (error) {
        message(claimMessage, `Could not load the claim form. ${error.message}`, true);
    }
}

initialize();
