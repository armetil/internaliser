import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate


def generate_sample_orders(n=20, seed=42, qty_min=1, qty_max=10):
    np.random.seed(seed)
    sides = np.random.choice(['BUY', 'SELL'], size=n)
    instruments = np.random.choice(['AAPL', 'MSFT', 'GOOG'], size=n)
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    times = [start_time + timedelta(minutes=i) for i in range(n)]
    quantities = np.random.randint(qty_min, qty_max, size=n)

    df = pd.DataFrame({
        'order_id': range(1, n + 1),
        'side': sides,
        'instrument': instruments,
        'time': times,
        'quantity': quantities
    })
    return df


def simulate_internalizer_advanced(orders_df, time_window=timedelta(minutes=2)):
    orders_df = orders_df.sort_values(by='time').reset_index(drop=True)
    n = len(orders_df)
    matched_quantity = [0] * n
    remaining_quantity = list(orders_df['quantity'])
    unmatched = defaultdict(deque)
    match_events = []

    def remove_expired_from_front(q: deque, current_time: datetime):
        while q:
            _, front_time, front_qty = q[0]
            if front_time < current_time - time_window or front_qty <= 0:
                q.popleft()
            else:
                break

    for i, row in orders_df.iterrows():
        current_time = row['time']
        current_side = row['side']
        instr = row['instrument']
        current_qty = remaining_quantity[i]
        opposite_side = 'SELL' if current_side == 'BUY' else 'BUY'
        opp_key = (instr, opposite_side)
        remove_expired_from_front(unmatched[opp_key], current_time)

        while current_qty > 0 and unmatched[opp_key]:
            old_idx, old_time, old_qty = unmatched[opp_key][0]
            if old_time < current_time - time_window or old_qty <= 0:
                unmatched[opp_key].popleft()
                continue
            matchable_qty = min(current_qty, old_qty)
            matched_quantity[i] += matchable_qty
            matched_quantity[old_idx] += matchable_qty
            match_events.append({
                'buy_order_id': i if current_side == 'BUY' else old_idx,
                'sell_order_id': old_idx if current_side == 'BUY' else i,
                'matched_quantity': matchable_qty,
                'match_time': current_time,
                'buy_time': row['time'] if current_side == 'BUY' else old_time,
                'sell_time': old_time if current_side == 'BUY' else row['time']
            })
            current_qty -= matchable_qty
            old_qty -= matchable_qty
            if old_qty == 0:
                unmatched[opp_key].popleft()
            else:
                unmatched[opp_key][0] = (old_idx, old_time, old_qty)

        remaining_quantity[i] = current_qty
        if current_qty > 0:
            my_key = (instr, current_side)
            unmatched[my_key].append((i, current_time, current_qty))

    result_df = orders_df.copy()
    result_df['matched_quantity'] = matched_quantity
    result_df['remaining_quantity'] = remaining_quantity

    match_df = pd.DataFrame(match_events)
    if not match_df.empty:
        match_df['buy_time_to_match'] = (match_df['match_time'] - match_df['buy_time']).dt.total_seconds()
        match_df['sell_time_to_match'] = (match_df['match_time'] - match_df['sell_time']).dt.total_seconds()
        buy_match_avg = match_df.groupby('buy_order_id')['buy_time_to_match'].mean().rename('avg_time_to_match')
        sell_match_avg = match_df.groupby('sell_order_id')['sell_time_to_match'].mean().rename('avg_time_to_match')
        time_to_match = pd.concat([buy_match_avg, sell_match_avg]).groupby(level=0).mean()
        result_df['avg_time_to_match'] = result_df.index.map(time_to_match)
    else:
        result_df['avg_time_to_match'] = np.nan

    return result_df


def compute_internalization_rate(matched_df):
    total_shares = matched_df['quantity'].sum()
    total_matched = matched_df['matched_quantity'].sum()
    return (total_matched / total_shares) if total_shares else 0.0


def compute_internalization_rate_by_instrument(matched_df):
    group = matched_df.groupby('instrument').agg(
        total_quantity=('quantity', 'sum'),
        total_matched=('matched_quantity', 'sum')
    )
    group['internalization_rate'] = group['total_matched'] / group['total_quantity']
    group['internalization_rate'] = (group['internalization_rate'] * 100).round(2).astype(str) + '%'
    return group


def compute_fill_rate_stats(matched_df):
    full_fill_rate = (matched_df['matched_quantity'] == matched_df['quantity']).mean()
    partial_fill_rate = ((matched_df['matched_quantity'] > 0) & (matched_df['matched_quantity'] < matched_df['quantity'])).mean()
    grouped = matched_df.groupby('instrument')
    fill_stats_by_instr = grouped.apply(lambda g: pd.Series({
        'full_fill_rate': (g['matched_quantity'] == g['quantity']).mean(),
        'partial_fill_rate': ((g['matched_quantity'] > 0) & (g['matched_quantity'] < g['quantity'])).mean()
    }))
    return full_fill_rate, partial_fill_rate, fill_stats_by_instr


def plot_combined_dashboard(matched_df, rolling_window=1000):
    instruments = matched_df['instrument'].unique()
    num_rows = 3 + len(instruments)  # 1 for cumulative, 1 for instrument cumulative, 1 for all avg_time, rest per instrument

    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=(
            ["Overall Internalization Rate"] +
            ["Cumulative & Rolling Rate by Instrument"] +
            ["Distribution of Avg Time to Match"] +
            [f"Avg Time to Match - {instr}" for instr in instruments]
        ),
        shared_xaxes=False
    )

    df_sorted = matched_df.sort_values('time').copy()
    df_sorted['cumulative_quantity'] = df_sorted['quantity'].cumsum()
    df_sorted['cumulative_matched'] = df_sorted['matched_quantity'].cumsum()
    df_sorted['cumulative_rate'] = df_sorted['cumulative_matched'] / df_sorted['cumulative_quantity']
    df_sorted['rolling_quantity'] = df_sorted['quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_matched'] = df_sorted['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum()
    df_sorted['rolling_rate'] = df_sorted['rolling_matched'] / df_sorted['rolling_quantity']

    # Overall
    fig.add_trace(go.Scatter(x=df_sorted['time'], y=df_sorted['cumulative_rate'], name='Cumulative Rate'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sorted['time'], y=df_sorted['rolling_rate'], name='Rolling Rate'), row=1, col=1)
    fig.update_yaxes(title_text='Rate', row=1, col=1)

    # Instrument cumulative & rolling
    for instr in sorted(instruments):
        subset = df_sorted[df_sorted['instrument'] == instr].copy()
        subset['cumulative_instr'] = subset['matched_quantity'].cumsum() / subset['quantity'].cumsum()
        subset['rolling_instr'] = (
            subset['matched_quantity'].rolling(window=rolling_window, min_periods=1).sum()
            / subset['quantity'].rolling(window=rolling_window, min_periods=1).sum()
        )
        fig.add_trace(go.Scatter(x=subset['time'], y=subset['cumulative_instr'], name=f'{instr} Cumulative'), row=2, col=1)
        fig.add_trace(go.Scatter(x=subset['time'], y=subset['rolling_instr'], name=f'{instr} Rolling'), row=2, col=1)
    fig.update_yaxes(title_text='Rate', row=2, col=1)

    # Avg time overall
    fig.add_trace(go.Histogram(x=matched_df['avg_time_to_match'].dropna(), name="All Avg Time to Match", nbinsx=100), row=3, col=1)

    # Avg time per instrument
    for i, instr in enumerate(instruments):
        x = matched_df[matched_df['instrument'] == instr]['avg_time_to_match'].dropna()
        fig.add_trace(go.Histogram(x=x, name=instr, nbinsx=100), row=4 + i, col=1)
        fig.update_xaxes(title_text='Time Sec', row=4 + i, col=1)
        fig.update_yaxes(title_text='Orders', row=4 + i, col=1)

    fig.update_layout(
        height=300 * num_rows,
        width=1000,
        title_text="Internalization Dashboard"
    )
    fig.show()


if __name__ == "__main__":
    df_orders = generate_sample_orders(n=2000, seed=42)
    matched_df = simulate_internalizer_advanced(df_orders, time_window=timedelta(minutes=2))
    matched_df['match_fill_rate'] = matched_df['matched_quantity'] / matched_df['quantity']
    matched_df['day'] = matched_df['time'].dt.date

    overall_rate = compute_internalization_rate(matched_df)
    rate_by_instr = compute_internalization_rate_by_instrument(matched_df)
    full_fill_rate, partial_fill_rate, fill_stats_by_instr = compute_fill_rate_stats(matched_df)

    overall_rate_df = pd.DataFrame([["Overall", overall_rate]], columns=["Type", "Internalization Rate"])
    rate_by_instr_df = rate_by_instr.reset_index()
    fill_stats_df = fill_stats_by_instr.reset_index()

    matched_df['day'] = matched_df['time'].dt.date
    # Group by instrument and day
    daily_summary = matched_df.groupby(['instrument', 'day']).agg(
        day_quantity=('quantity', 'sum'),
        day_matched=('matched_quantity', 'sum')
    )

    desc_df = (
        matched_df.reset_index().groupby([matched_df.time.dt.date]).agg(
            day_quantity=('quantity', 'sum'),
            day_matched=('matched_quantity', 'sum')))
    print(desc_df.describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(1))

    # Describe per instrument
    print("\n=== Description of Daily Quantity/Matched by Instrument ===")
    for instr in daily_summary.index.get_level_values(0).unique():
        print(f"\n--- Instrument: {instr} ---")
        desc = daily_summary.loc[instr].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(2)
        print(desc)


    # Compute match fill rate per order
    matched_df['match_fill_rate'] = matched_df['matched_quantity'] / matched_df['quantity']

    # Summary describe of match fill rate per order
    print("\n=== Match Fill Rate per Order (All Instruments) ===")
    print(matched_df['match_fill_rate'].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(3))

    # Optionally: describe fill rate per instrument
    print("\n=== Match Fill Rate per Order by Instrument ===")
    for instr in matched_df['instrument'].unique():
        subset = matched_df[matched_df['instrument'] == instr]
        print(f"\n--- Instrument: {instr} ---")
        print(subset['match_fill_rate'].describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).round(3))

    overall_html = overall_rate_df.style.format({"Internalization Rate": "{:.2%}"}).to_html(index=False)
    rate_by_instr_html = rate_by_instr_df.to_html(index=False)
    fill_stats_html = fill_stats_df.style.format({"full_fill_rate": "{:.2%}", "partial_fill_rate": "{:.2%}"}).to_html(index=False)

    combined_html = f"""
           <html>
           <head><title>Internalization Report</title></head>
           <body>
           <h2>Overall Internalization Rate</h2>
           {overall_html}
    
           <h2>Internalization Rate by Instrument</h2>
           {rate_by_instr_html}
    
           <h2>Fill Rates by Instrument</h2>
           {fill_stats_html}
           </body>
           </html>
           """

    with open("internalization_report.html", "w") as f:
        f.write(combined_html)
    print("Internalization metrics exported to 'internalization_report.html'")

    plot_combined_dashboard(matched_df)
