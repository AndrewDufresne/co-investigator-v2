"""Case viewer component — displays structured case data in organized panels."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_subject_card(subject: dict[str, Any]) -> None:
    """Render subject profile card."""
    st.subheader("👤 Subject Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {subject.get('name', 'N/A')}")
        st.write(f"**DOB:** {subject.get('dob', 'N/A')}")
        st.write(f"**Occupation:** {subject.get('occupation', 'N/A')}")
        st.write(f"**Customer Since:** {subject.get('customer_since', 'N/A')}")
    with col2:
        risk = subject.get("risk_rating", "unknown")
        color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
        st.write(f"**Risk Rating:** {color} {risk.upper()}")
        st.write(f"**Address:** {subject.get('address', 'N/A')}")
        st.write(f"**Phone:** {subject.get('phone', 'N/A')}")
        st.write(f"**Email:** {subject.get('email', 'N/A')}")


def render_account_table(accounts: list[dict[str, Any]]) -> None:
    """Render accounts summary table."""
    st.subheader("🏦 Accounts")
    if not accounts:
        st.info("No account data available.")
        return

    for acc in accounts:
        with st.container():
            cols = st.columns(4)
            cols[0].metric("Account", acc.get("account_id", "N/A"))
            cols[1].metric("Type", acc.get("account_type", "N/A"))
            cols[2].metric("Balance", f"${acc.get('balance', 0):,.2f}")
            cols[3].metric("Currency", acc.get("currency", "USD"))


def render_transaction_table(transactions: list[dict[str, Any]], summary: dict | None = None) -> None:
    """Render transaction list with summary stats."""
    st.subheader("💳 Transactions")

    if summary:
        cols = st.columns(4)
        cols[0].metric("Total Count", summary.get("count", 0))
        cols[1].metric("Total Inflow", f"${summary.get('total_inflow', 0):,.2f}")
        cols[2].metric("Total Outflow", f"${summary.get('total_outflow', 0):,.2f}")
        date_range = summary.get("date_range", {})
        cols[3].metric("Date Range", f"{date_range.get('start', '?')} → {date_range.get('end', '?')}")

    if not transactions:
        st.info("No transaction data.")
        return

    # Build display table
    display_data = []
    for t in transactions:
        flags = ", ".join(t.get("risk_flags", []))
        display_data.append({
            "ID": t.get("txn_id", ""),
            "Date": t.get("date", ""),
            "Type": t.get("type", ""),
            "Amount": f"${t.get('amount', 0):,.2f}",
            "From": t.get("from_entity", t.get("from_account", "")),
            "To": t.get("to_entity", t.get("to_account", "")),
            "Country": f"{t.get('from_country', '')}→{t.get('to_country', '')}",
            "Risk Flags": flags or "—",
        })

    st.dataframe(display_data, width='stretch', hide_index=True)


def render_case_overview(data: dict[str, Any]) -> None:
    """Render full case overview with all sections."""
    st.header(f"📋 Case: {data.get('case_id', 'N/A')}")
    st.caption(f"Alert Date: {data.get('alert_date', 'N/A')} | Priority: {data.get('priority', 'N/A')}")

    with st.expander("Subject Profile", expanded=True):
        render_subject_card(data.get("subject", {}))

    with st.expander("Accounts"):
        render_account_table(data.get("accounts", []))

    with st.expander("Transactions", expanded=True):
        render_transaction_table(
            data.get("transactions", []),
            data.get("transaction_summary"),
        )

    with st.expander("KYC Summary"):
        kyc = data.get("kyc", {})
        if kyc:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Verification:** {kyc.get('verification_status', 'N/A')}")
                st.write(f"**Source of Funds:** {kyc.get('source_of_funds', 'N/A')}")
                st.write(f"**PEP Status:** {'⚠️ Yes' if kyc.get('pep_status') else '✅ No'}")
            with col2:
                st.write(f"**Expected Activity:** {kyc.get('expected_activity', 'N/A')}")
                st.write(f"**Actual Activity:** {kyc.get('actual_activity_profile', 'N/A')}")
                mismatch = kyc.get("activity_mismatch", False)
                st.write(f"**Mismatch:** {'🔴 Yes' if mismatch else '🟢 No'}")

    if data.get("alerts"):
        with st.expander("Alerts"):
            for a in data["alerts"]:
                sev = a.get("severity", "unknown")
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(sev, "🔵")
                st.write(f"{icon} **[{a.get('alert_id')}]** {a.get('type')} — {a.get('description')}")

    if data.get("related_entities"):
        with st.expander("Related Entities"):
            for e in data["related_entities"]:
                st.write(
                    f"• **{e.get('entity_name')}** ({e.get('entity_type')}) — "
                    f"{e.get('jurisdiction')} — {e.get('relationship')}"
                )
                if e.get("risk_notes"):
                    st.caption(f"  ⚠️ {e['risk_notes']}")
